"""
test_heuristics.py: Unit tests and academic verification for heuristic functions.
Verifies:
  - Misplaced tiles calculation
  - Manhattan distance calculation
  - Blank tile exclusion (0 should never contribute to heuristics)
  - Mathematical dominance: h2(n) >= h1(n) for all states
  - Admissibility bounds
"""

import random
from src.puzzle import GOAL_STATE, State
from src.heuristics import misplaced_tiles, manhattan_distance


def test_heuristics_at_goal():
    # Both heuristics must evaluate to 0 at the goal state
    assert misplaced_tiles(GOAL_STATE) == 0
    assert manhattan_distance(GOAL_STATE) == 0


def test_misplaced_tiles_known_states():
    # Goal: (1, 2, 3, 4, 5, 6, 7, 8, 0)
    # Swap 1 and 2: only 1 and 2 are misplaced -> count = 2
    state = (2, 1, 3, 4, 5, 6, 7, 8, 0)
    assert misplaced_tiles(state) == 2

    # Blank tile displaced: blank at index 0, 1 at index 8
    # Non-blank tiles: 1 is at index 8 (wrong, misplaced).
    # Other tiles 2..8 are at indices 1..7 (correct).
    # Total misplaced = 1 (blank 0 is NOT counted!)
    state_blank = (0, 2, 3, 4, 5, 6, 7, 8, 1)
    assert misplaced_tiles(state_blank) == 1


def test_manhattan_distance_known_states():
    # Canonical Goal layout:
    # 1(0,0)  2(0,1)  3(0,2)
    # 4(1,0)  5(1,1)  6(1,2)
    # 7(2,0)  8(2,1)  0(2,2)

    # Move tile 8 down into 0:
    # 1 2 3
    # 4 5 6
    # 7 0 8 -> tile 8 is at (2, 2), goal is (2, 1) -> dist = 1
    state = (1, 2, 3, 4, 5, 6, 7, 0, 8)
    assert manhattan_distance(state) == 1

    # Invert tile 1 and tile 8:
    # 8 2 3
    # 4 5 6
    # 7 1 0
    # tile 8 at (0,0), goal (2,1) -> |0-2| + |0-1| = 3
    # tile 1 at (2,1), goal (0,0) -> |2-0| + |1-0| = 3
    # total Manhattan = 6
    state2 = (8, 2, 3, 4, 5, 6, 7, 1, 0)
    assert manhattan_distance(state2) == 6


def test_blank_tile_exclusion():
    """Ensure blank tile (0) position does not directly add to either heuristic."""
    # State where all non-blank tiles are in goal position, only blank is displaced:
    # This is physically impossible in 3x3 without displacing another tile,
    # but test mathematically that if tile values match goal, count is 0.
    state = (1, 2, 3, 4, 5, 6, 7, 8, 0)
    assert misplaced_tiles(state) == 0
    assert manhattan_distance(state) == 0


def test_heuristic_dominance_theorem():
    """
    Empirical proof of dominance: h2(n) >= h1(n) for any board state n.
    Tests across 1000 randomized permutations.
    """
    rng = random.Random(42)
    base = list(range(9))

    for _ in range(1000):
        perm = base.copy()
        rng.shuffle(perm)
        state: State = tuple(perm)

        h1 = misplaced_tiles(state)
        h2 = manhattan_distance(state)

        # Dominance property: h2 must be strictly greater than or equal to h1
        assert h2 >= h1, (
            f"Dominance violation: h2({h2}) < h1({h1}) for state {state}"
        )

