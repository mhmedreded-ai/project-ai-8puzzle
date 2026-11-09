"""
puzzle.py: Core 8-Puzzle Problem Formulation and Representation.

Academic 5-Tuple Problem Formulation:
--------------------------------------
1. Initial State: Any valid permutation of {0, 1, ..., 8} in a 3x3 grid,
   represented as a 9-element tuple of integers where 0 denotes the blank tile.
2. Actions: {'UP', 'DOWN', 'LEFT', 'RIGHT'}, defined as the directional movement
   of the blank tile (0).
3. Transition Model: Result(s, a) returns the new state resulting from moving the
   blank tile in the specified action direction.
4. Goal Test: Checks if the current state strictly matches GOAL_STATE:
   (1, 2, 3, 4, 5, 6, 7, 8, 0).
5. Path Cost: Step cost c(s, a, s') = 1 for every transition. The total path
   cost is equal to the depth/number of moves g(n).
"""

from __future__ import annotations
from typing import Optional, Sequence, Generator
import itertools

# Standard Canonical 8-Puzzle Goal State
# 1 2 3
# 4 5 6
# 7 8 0 (blank is at bottom-right, index 8)
GOAL_STATE: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 0)

State = tuple[int, ...]

# Global counter for tie-breaking in search priority queues
_node_counter = itertools.count()


class Node:
    """
    Search tree node encapsulating the state and search metadata.

    Attributes:
        state: Board state represented as a 9-element tuple.
        parent: Pointer to the predecessor Node (None for the root).
        action: The action taken from parent to reach this state.
        g: Path cost from the initial state to this node (depth in unit-cost search).
        h: Heuristic estimate from this node's state to the goal state.
        f: Evaluation function value (f = g + h for A*, f = h for Greedy).
        depth: Tree depth of this node.
        creation_id: Unique insertion order identifier used for stable tie-breaking.
    """

    __slots__ = ("state", "parent", "action", "g", "h", "f", "depth", "creation_id")

    def __init__(
        self,
        state: State,
        parent: Optional[Node] = None,
        action: Optional[str] = None,
        g: int = 0,
        h: float = 0.0,
        f: Optional[float] = None,
    ) -> None:
        self.state: State = state
        self.parent: Optional[Node] = parent
        self.action: Optional[str] = action
        self.g: int = g
        self.h: float = h
        self.f: float = f if f is not None else (g + h)
        self.depth: int = parent.depth + 1 if parent is not None else 0
        self.creation_id: int = next(_node_counter)

    def __lt__(self, other: Node) -> bool:
        """
        Priority comparison for min-heap priority queues (heapq).
        Academic tie-breaking strategy:
        1. Compare primary evaluation value f.
        2. In case of tie, prefer node with lower h (closer to goal).
        3. In case of further tie, break deterministically using insertion order.
        """
        if self.f != other.f:
            return self.f < other.f
        if self.h != other.h:
            return self.h < other.h
        return self.creation_id < other.creation_id

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        return self.state == other.state

    def __hash__(self) -> int:
        return hash(self.state)

    def __repr__(self) -> str:
        return (
            f"Node(state={self.state}, action={self.action}, "
            f"g={self.g}, h={self.h:.2f}, f={self.f:.2f})"
        )

    def extract_path(self) -> list[Node]:
        """
        Reconstructs the solution path from the initial state node to this node.

        Returns:
            list[Node]: Ordered list of nodes along the solution trajectory.
        """
        path: list[Node] = []
        curr: Optional[Node] = self
        while curr is not None:
            path.append(curr)
            curr = curr.parent
        path.reverse()
        return path


# Direction vectors for blank movement: (row_delta, col_delta)
DIRECTIONS: dict[str, tuple[int, int]] = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}


def apply_action(state: State, action: str) -> State:
    """
    Transition Model: Result(s, a).

    Executes a blank-tile movement action on the state and returns the new state.

    Args:
        state: Current 9-tuple board state.
        action: One of 'UP', 'DOWN', 'LEFT', 'RIGHT'.

    Returns:
        State: The resulting 9-tuple board state after the move.

    Raises:
        ValueError: If action is invalid or physically illegal from the current blank position.
    """
    if action not in DIRECTIONS:
        raise ValueError(f"Unknown action: {action}. Must be one of {list(DIRECTIONS.keys())}")

    blank_idx = state.index(0)
    blank_r, blank_c = divmod(blank_idx, 3)
    dr, dc = DIRECTIONS[action]
    new_r, new_c = blank_r + dr, blank_c + dc

    if not (0 <= new_r < 3 and 0 <= new_c < 3):
        raise ValueError(f"Action '{action}' is illegal from blank position ({blank_r}, {blank_c}).")

    new_blank_idx = new_r * 3 + new_c
    state_list = list(state)
    # Swap blank tile (0) with the target adjacent tile
    state_list[blank_idx], state_list[new_blank_idx] = (
        state_list[new_blank_idx],
        state_list[blank_idx],
    )
    return tuple(state_list)


def get_successors(state: State) -> list[tuple[str, State]]:
    """
    Generates all legal (action, next_state) successor pairs from the given state.

    In the 8-puzzle, successors correspond to moving the blank tile (0)
    UP, DOWN, LEFT, or RIGHT within the 3x3 boundaries.

    Args:
        state: Current 9-tuple state.

    Returns:
        list[tuple[str, State]]: Valid (action_name, resulting_state) pairs.
    """
    blank_idx = state.index(0)
    blank_r, blank_c = divmod(blank_idx, 3)
    successors: list[tuple[str, State]] = []

    # Try all 4 candidate directions
    for action, (dr, dc) in DIRECTIONS.items():
        nr, nc = blank_r + dr, blank_c + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            target_idx = nr * 3 + nc
            s_list = list(state)
            s_list[blank_idx], s_list[target_idx] = s_list[target_idx], s_list[blank_idx]
            successors.append((action, tuple(s_list)))

    return successors


def count_inversions(state: Sequence[int]) -> int:
    """
    Counts the number of inversions in the state permutation, ignoring the blank tile (0).

    An inversion is a pair of tiles (a, b) such that a appears before b in the
    linearized board representation, but a > b.

    Args:
        state: Sequence of 9 integers representing the 3x3 board.

    Returns:
        int: Total number of inversions.
    """
    tiles = [tile for tile in state if tile != 0]
    inversions = 0
    n = len(tiles)
    for i in range(n):
        for j in range(i + 1, n):
            if tiles[i] > tiles[j]:
                inversions += 1
    return inversions


def is_solvable(state: State, goal: State = GOAL_STATE) -> bool:
    """
    Determines whether a given 8-puzzle state can reach the goal state.

    Academic Invariant Proof:
    -------------------------
    In an N x N puzzle where N is odd (here N = 3):
    1. A horizontal move of the blank tile slides a tile left or right within
       the same row. In the linearized array, this changes the position of the
       blank tile by +/- 1, leaving the relative ordering of all non-blank
       tiles completely unchanged. Hence, inversion count delta = 0.
    2. A vertical move shifts the blank tile by +/- 3 positions in the array.
       This moves one tile over exactly two other non-blank tiles. Passing over
       two tiles changes the inversion count by either +2, -2, or 0.
    3. In all legal moves, the inversion count changes strictly by an even number:
       Delta(inversions) = 0 (mod 2).
    4. Therefore, the parity (inversion count mod 2) is an invariant under all
       valid state transitions.
    5. The canonical goal state (1, 2, 3, 4, 5, 6, 7, 8, 0) has 0 inversions
       (even parity). Therefore, an initial state is solvable if and only if
       count_inversions(state) is EVEN.

    Args:
        state: Initial 9-tuple state to test.
        goal: Target goal state (defaults to canonical GOAL_STATE).

    Returns:
        bool: True if reachable, False if unsolvable.
    """
    state_inversions = count_inversions(state)
    goal_inversions = count_inversions(goal)
    return (state_inversions % 2) == (goal_inversions % 2)


def format_board(state: State) -> str:
    """
    Renders an 8-puzzle state into a clean Unicode/ASCII box format.

    Args:
        state: 9-tuple board state.

    Returns:
        str: Formatted multi-line string representing the 3x3 grid.
    """
    symbols = [str(x) if x != 0 else " " for x in state]
    lines = [
        "┌───┬───┬───┐",
        f"│ {symbols[0]} │ {symbols[1]} │ {symbols[2]} │",
        "├───┼───┼───┤",
        f"│ {symbols[3]} │ {symbols[4]} │ {symbols[5]} │",
        "├───┼───┼───┤",
        f"│ {symbols[6]} │ {symbols[7]} │ {symbols[8]} │",
        "└───┴───┴───┘",
    ]
    return "\n".join(lines)


def parse_board(raw_input: str | Sequence[int]) -> State:
    """
    Parses and validates user input into a canonical 9-tuple state.

    Accepts:
        - String of 9 whitespace or comma-separated integers (e.g. "1 2 3 4 5 6 7 8 0").
        - Flat list/tuple of 9 integers.
        - 3x3 2D list of integers.

    Returns:
        State: Validated 9-tuple.

    Raises:
        ValueError: If input does not contain exactly the digits 0 through 8.
    """
    flat: list[int] = []
    if isinstance(raw_input, str):
        # Remove brackets, commas
        cleaned = raw_input.replace("[", " ").replace("]", " ").replace(",", " ")
        tokens = cleaned.strip().split()
        try:
            flat = [int(tok) for tok in tokens]
        except ValueError as exc:
            raise ValueError(f"All board entries must be integers: {exc}") from exc
    elif isinstance(raw_input, (list, tuple)):
        for item in raw_input:
            if isinstance(item, (list, tuple)):
                flat.extend(int(x) for x in item)
            else:
                flat.append(int(item))
    else:
        raise TypeError(f"Unsupported board input type: {type(raw_input)}")

    if len(flat) != 9:
        raise ValueError(f"Board must contain exactly 9 tiles, got {len(flat)}: {flat}")

    if sorted(flat) != list(range(9)):
        raise ValueError(
            f"Board must be a permutation of digits 0 through 8. Received: {flat}"
        )

    return tuple(flat)

