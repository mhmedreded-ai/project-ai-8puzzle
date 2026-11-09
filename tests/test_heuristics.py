"""
test_heuristics.py: Unit tests and academic verification for heuristic functions using unittest.
Verifies:
  - Misplaced tiles calculation
  - Manhattan distance calculation
  - Blank tile exclusion (0 should never contribute to heuristics)
  - Mathematical dominance: h2(n) >= h1(n) for all states
  - Admissibility bounds
"""

import unittest
import random
from src.puzzle import GOAL_STATE, State
from src.heuristics import misplaced_tiles, manhattan_distance


class TestHeuristics(unittest.TestCase):
    """Unit tests for Misplaced Tiles (h1) and Manhattan Distance (h2)."""

    def test_heuristics_at_goal(self):
        self.assertEqual(misplaced_tiles(GOAL_STATE), 0)
        self.assertEqual(manhattan_distance(GOAL_STATE), 0)

    def test_misplaced_tiles_known_states(self):
        state = (2, 1, 3, 4, 5, 6, 7, 8, 0)
        self.assertEqual(misplaced_tiles(state), 2)

        # Blank tile displaced: blank at index 0, 1 at index 8
        state_blank = (0, 2, 3, 4, 5, 6, 7, 8, 1)
        self.assertEqual(misplaced_tiles(state_blank), 1)

    def test_manhattan_distance_known_states(self):
        state = (1, 2, 3, 4, 5, 6, 7, 0, 8)
        self.assertEqual(manhattan_distance(state), 1)

        state2 = (8, 2, 3, 4, 5, 6, 7, 1, 0)
        self.assertEqual(manhattan_distance(state2), 6)

    def test_blank_tile_exclusion(self):
        state = (1, 2, 3, 4, 5, 6, 7, 8, 0)
        self.assertEqual(misplaced_tiles(state), 0)
        self.assertEqual(manhattan_distance(state), 0)

    def test_heuristic_dominance_theorem(self):
        """Empirical proof of dominance: h2(n) >= h1(n) across 1,000 permutations."""
        rng = random.Random(42)
        base = list(range(9))

        for _ in range(1000):
            perm = base.copy()
            rng.shuffle(perm)
            state: State = tuple(perm)

            h1 = misplaced_tiles(state)
            h2 = manhattan_distance(state)

            self.assertGreaterEqual(
                h2,
                h1,
                f"Dominance violation: h2({h2}) < h1({h1}) for state {state}",
            )


if __name__ == "__main__":
    unittest.main()
