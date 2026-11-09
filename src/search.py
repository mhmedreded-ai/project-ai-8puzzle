"""
search.py: Search Algorithms for the 8-Puzzle.

Implements:
1. A* Search (Informed, Optimal & Complete with consistent heuristics: f(n) = g(n) + h(n))
2. Greedy Best-First Search (Informed, Non-optimal: f(n) = h(n))
3. Breadth-First Search (Uninformed, Optimal for unit step costs, Complete)

All search routines track academic search metrics:
- Solution path (sequence of Nodes / states)
- Path cost g(n) (solution depth)
- Number of nodes expanded
- Number of nodes generated
- Maximum frontier size (space complexity)
- Execution time in milliseconds
"""

from __future__ import annotations
import collections
import heapq
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Iterator, Any

from src.puzzle import Node, State, GOAL_STATE, get_successors, is_solvable
from src.heuristics import HeuristicFn, misplaced_tiles, manhattan_distance


@dataclass
class SearchResult:
    """
    Encapsulates the comprehensive outcome and performance metrics of a search run.

    Supports tuple unpacking:
        solution_path, nodes_expanded, elapsed_time = search_fn(...)
    """

    algorithm: str
    heuristic_name: Optional[str]
    success: bool
    solution_path: list[Node] = field(default_factory=list)
    cost: int = 0
    nodes_expanded: int = 0
    nodes_generated: int = 0
    max_frontier_size: int = 0
    elapsed_time: float = 0.0  # In milliseconds

    @property
    def elapsed_time_ms(self) -> float:
        """Alias for elapsed execution time in milliseconds."""
        return self.elapsed_time

    @property
    def actions(self) -> list[str]:
        """Returns the list of actions taken to reach the goal."""
        return [node.action for node in self.solution_path[1:] if node.action is not None]

    @property
    def states(self) -> list[State]:
        """Returns the sequence of board states from root to goal."""
        return [node.state for node in self.solution_path]

    def __iter__(self) -> Iterator[Any]:
        """Allows unpacking into (solution_path, nodes_expanded, elapsed_time)."""
        yield self.solution_path
        yield self.nodes_expanded
        yield self.elapsed_time


def a_star_search(
    initial_state: State,
    heuristic: Optional[HeuristicFn] = None,
    goal_state: State = GOAL_STATE,
) -> SearchResult:
    """
    A* Graph Search Algorithm.

    Evaluation Function:
        f(n) = g(n) + h(n)
        where:
          - g(n) is the exact cost from start to node n.
          - h(n) is the heuristic estimate from node n to the goal.

    Optimality & Completeness:
        Since both misplaced_tiles and manhattan_distance are consistent (monotone),
        A* graph search is guaranteed to return an optimal solution (minimum path cost)
        without having to re-open nodes in the closed set.

    Args:
        initial_state: 9-tuple starting board.
        heuristic: Heuristic function h(state, goal). Defaults to manhattan_distance.
        goal_state: 9-tuple target board (defaults to canonical GOAL_STATE).

    Returns:
        SearchResult: Contains solution path, nodes expanded, execution time, etc.
    """
    h_fn = heuristic if heuristic is not None else manhattan_distance
    heuristic_name = h_fn.__name__ if hasattr(h_fn, "__name__") else "Custom Heuristic"

    start_time = time.perf_counter()

    # Pre-check solvability invariant
    if not is_solvable(initial_state, goal_state):
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return SearchResult(
            algorithm="A* Search",
            heuristic_name=heuristic_name,
            success=False,
            solution_path=[],
            cost=-1,
            nodes_expanded=0,
            nodes_generated=0,
            max_frontier_size=0,
            elapsed_time=elapsed_ms,
        )

    # Initial root node
    initial_h = h_fn(initial_state, goal_state)
    root = Node(
        state=initial_state,
        parent=None,
        action=None,
        g=0,
        h=float(initial_h),
        f=float(initial_h),
    )

    # Frontier min-heap
    frontier: list[Node] = [root]
    heapq.heapify(frontier)

    # Best known cost g(s) to reach each state
    cost_so_far: dict[State, int] = {initial_state: 0}
    explored: set[State] = set()

    nodes_expanded = 0
    nodes_generated = 1
    max_frontier_size = 1

    while frontier:
        max_frontier_size = max(max_frontier_size, len(frontier))
        current_node = heapq.heappop(frontier)

        # Goal test upon expansion (standard A* requirement for optimality)
        if current_node.state == goal_state:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            path = current_node.extract_path()
            return SearchResult(
                algorithm="A* Search",
                heuristic_name=heuristic_name,
                success=True,
                solution_path=path,
                cost=current_node.g,
                nodes_expanded=nodes_expanded,
                nodes_generated=nodes_generated,
                max_frontier_size=max_frontier_size,
                elapsed_time=elapsed_ms,
            )

        # Skip stale entries in priority queue
        if current_node.state in explored:
            continue

        explored.add(current_node.state)
        nodes_expanded += 1

        for action, next_state in get_successors(current_node.state):
            new_g = current_node.g + 1

            if next_state not in cost_so_far or new_g < cost_so_far[next_state]:
                cost_so_far[next_state] = new_g
                h_val = float(h_fn(next_state, goal_state))
                child_node = Node(
                    state=next_state,
                    parent=current_node,
                    action=action,
                    g=new_g,
                    h=h_val,
                    f=new_g + h_val,
                )
                nodes_generated += 1
                heapq.heappush(frontier, child_node)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return SearchResult(
        algorithm="A* Search",
        heuristic_name=heuristic_name,
        success=False,
        solution_path=[],
        cost=-1,
        nodes_expanded=nodes_expanded,
        nodes_generated=nodes_generated,
        max_frontier_size=max_frontier_size,
        elapsed_time=elapsed_ms,
    )


def greedy_search(
    initial_state: State,
    heuristic: Optional[HeuristicFn] = None,
    goal_state: State = GOAL_STATE,
) -> SearchResult:
    """
    Greedy Best-First Search Algorithm.

    Evaluation Function:
        f(n) = h(n)
        Expands the node that appears closest to the goal according to the heuristic.

    Properties:
        - Incomplete on infinite state spaces (complete on finite graphs with cycle checking).
        - Non-optimal: Frequently finds suboptimal solutions because it ignores the path
          cost g(n) accumulated so far.

    Args:
        initial_state: 9-tuple starting board.
        heuristic: Heuristic function h(state, goal). Defaults to manhattan_distance.
        goal_state: 9-tuple target board (defaults to canonical GOAL_STATE).

    Returns:
        SearchResult: Contains solution path, nodes expanded, execution time, etc.
    """
    h_fn = heuristic if heuristic is not None else manhattan_distance
    heuristic_name = h_fn.__name__ if hasattr(h_fn, "__name__") else "Custom Heuristic"

    start_time = time.perf_counter()

    if not is_solvable(initial_state, goal_state):
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return SearchResult(
            algorithm="Greedy Best-First",
            heuristic_name=heuristic_name,
            success=False,
            solution_path=[],
            cost=-1,
            nodes_expanded=0,
            nodes_generated=0,
            max_frontier_size=0,
            elapsed_time=elapsed_ms,
        )

    initial_h = float(h_fn(initial_state, goal_state))
    root = Node(
        state=initial_state,
        parent=None,
        action=None,
        g=0,
        h=initial_h,
        f=initial_h,  # In Greedy, f(n) = h(n)
    )

    frontier: list[Node] = [root]
    heapq.heapify(frontier)

    explored: set[State] = set()
    in_frontier: set[State] = {initial_state}

    nodes_expanded = 0
    nodes_generated = 1
    max_frontier_size = 1

    while frontier:
        max_frontier_size = max(max_frontier_size, len(frontier))
        current_node = heapq.heappop(frontier)

        if current_node.state == goal_state:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            path = current_node.extract_path()
            return SearchResult(
                algorithm="Greedy Best-First",
                heuristic_name=heuristic_name,
                success=True,
                solution_path=path,
                cost=current_node.g,
                nodes_expanded=nodes_expanded,
                nodes_generated=nodes_generated,
                max_frontier_size=max_frontier_size,
                elapsed_time=elapsed_ms,
            )

        if current_node.state in explored:
            continue

        explored.add(current_node.state)
        nodes_expanded += 1

        for action, next_state in get_successors(current_node.state):
            if next_state not in explored and next_state not in in_frontier:
                h_val = float(h_fn(next_state, goal_state))
                child_node = Node(
                    state=next_state,
                    parent=current_node,
                    action=action,
                    g=current_node.g + 1,
                    h=h_val,
                    f=h_val,
                )
                nodes_generated += 1
                heapq.heappush(frontier, child_node)
                in_frontier.add(next_state)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return SearchResult(
        algorithm="Greedy Best-First",
        heuristic_name=heuristic_name,
        success=False,
        solution_path=[],
        cost=-1,
        nodes_expanded=nodes_expanded,
        nodes_generated=nodes_generated,
        max_frontier_size=max_frontier_size,
        elapsed_time=elapsed_ms,
    )


def bfs_search(
    initial_state: State,
    goal_state: State = GOAL_STATE,
) -> SearchResult:
    """
    Breadth-First Search (BFS) Graph Search Algorithm.

    Uninformed search baseline exploring the state space layer by layer.

    Properties:
        - Complete: Guaranteed to find a solution if one exists.
        - Optimal: Guaranteed to find the shallowest / shortest solution
          when all step costs are equal (c = 1).
        - High Space & Time Complexity: O(b^d) where b is branching factor (~3)
          and d is solution depth.

    Implementation:
        Uses an early goal test upon generation (standard AIMA optimization)
        and collections.deque for O(1) FIFO operations.

    Args:
        initial_state: 9-tuple starting board.
        goal_state: 9-tuple target board (defaults to canonical GOAL_STATE).

    Returns:
        SearchResult: Contains solution path, nodes expanded, execution time, etc.
    """
    start_time = time.perf_counter()

    if not is_solvable(initial_state, goal_state):
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return SearchResult(
            algorithm="BFS",
            heuristic_name=None,
            success=False,
            solution_path=[],
            cost=-1,
            nodes_expanded=0,
            nodes_generated=0,
            max_frontier_size=0,
            elapsed_time=elapsed_ms,
        )

    # Immediate root check
    root = Node(state=initial_state, parent=None, action=None, g=0, h=0.0, f=0.0)
    if initial_state == goal_state:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return SearchResult(
            algorithm="BFS",
            heuristic_name=None,
            success=True,
            solution_path=[root],
            cost=0,
            nodes_expanded=0,
            nodes_generated=1,
            max_frontier_size=1,
            elapsed_time=elapsed_ms,
        )

    frontier: collections.deque[Node] = collections.deque([root])
    reached: set[State] = {initial_state}

    nodes_expanded = 0
    nodes_generated = 1
    max_frontier_size = 1

    while frontier:
        max_frontier_size = max(max_frontier_size, len(frontier))
        current_node = frontier.popleft()
        nodes_expanded += 1

        for action, next_state in get_successors(current_node.state):
            if next_state not in reached:
                child_node = Node(
                    state=next_state,
                    parent=current_node,
                    action=action,
                    g=current_node.g + 1,
                    h=0.0,
                    f=float(current_node.g + 1),
                )
                nodes_generated += 1

                # Early goal test upon generation (Russell & Norvig Section 3.4.1)
                if next_state == goal_state:
                    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                    path = child_node.extract_path()
                    return SearchResult(
                        algorithm="BFS",
                        heuristic_name=None,
                        success=True,
                        solution_path=path,
                        cost=child_node.g,
                        nodes_expanded=nodes_expanded,
                        nodes_generated=nodes_generated,
                        max_frontier_size=max_frontier_size,
                        elapsed_time=elapsed_ms,
                    )

                reached.add(next_state)
                frontier.append(child_node)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return SearchResult(
        algorithm="BFS",
        heuristic_name=None,
        success=False,
        solution_path=[],
        cost=-1,
        nodes_expanded=nodes_expanded,
        nodes_generated=nodes_generated,
        max_frontier_size=max_frontier_size,
        elapsed_time=elapsed_ms,
    )

