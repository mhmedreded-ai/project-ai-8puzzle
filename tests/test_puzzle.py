"""
test_puzzle.py: Unit tests for core puzzle representation, state transitions,
inversion counts, solvability checks, and node mechanics using unittest.
"""

import unittest
from src.puzzle import (
    GOAL_STATE,
    Node,
    apply_action,
    get_successors,
    count_inversions,
    is_solvable,
    format_board,
    parse_board,
)


class TestPuzzleCore(unittest.TestCase):
    """Unit tests for 8-Puzzle board representation, actions, and solvability."""

    def test_goal_state_properties(self):
        self.assertEqual(GOAL_STATE, (1, 2, 3, 4, 5, 6, 7, 8, 0))
        self.assertEqual(len(GOAL_STATE), 9)
        self.assertEqual(count_inversions(GOAL_STATE), 0)
        self.assertTrue(is_solvable(GOAL_STATE))

    def test_node_ordering_and_tie_breaking(self):
        # Primary tie-breaking: lower f
        n1 = Node(state=GOAL_STATE, g=1, h=2, f=3)
        n2 = Node(state=GOAL_STATE, g=2, h=3, f=5)
        self.assertLess(n1, n2)

        # Secondary tie-breaking: same f, lower h preferred (closer to goal)
        n3 = Node(state=GOAL_STATE, g=3, h=1, f=4)
        n4 = Node(state=GOAL_STATE, g=2, h=2, f=4)
        self.assertLess(n3, n4)

        # Tertiary tie-breaking: insertion order (creation_id)
        n5 = Node(state=GOAL_STATE, g=2, h=2, f=4)
        self.assertLess(n4, n5)

    def test_node_extract_path(self):
        root = Node(state=(1, 2, 3, 4, 5, 6, 7, 0, 8), g=0)
        child = Node(state=GOAL_STATE, parent=root, action="RIGHT", g=1)

        path = child.extract_path()
        self.assertEqual(len(path), 2)
        self.assertEqual(path[0], root)
        self.assertEqual(path[1], child)
        self.assertEqual(path[1].action, "RIGHT")

    def test_apply_action_valid(self):
        state = (1, 2, 3, 4, 0, 5, 6, 7, 8)

        up_state = apply_action(state, "UP")
        self.assertEqual(up_state, (1, 0, 3, 4, 2, 5, 6, 7, 8))

        down_state = apply_action(state, "DOWN")
        self.assertEqual(down_state, (1, 2, 3, 4, 7, 5, 6, 0, 8))

        left_state = apply_action(state, "LEFT")
        self.assertEqual(left_state, (1, 2, 3, 0, 4, 5, 6, 7, 8))

        right_state = apply_action(state, "RIGHT")
        self.assertEqual(right_state, (1, 2, 3, 4, 5, 0, 6, 7, 8))

    def test_apply_action_invalid_boundary(self):
        corner_state = (0, 1, 2, 3, 4, 5, 6, 7, 8)

        with self.assertRaises(ValueError):
            apply_action(corner_state, "UP")

        with self.assertRaises(ValueError):
            apply_action(corner_state, "LEFT")

        self.assertEqual(apply_action(corner_state, "DOWN"), (3, 1, 2, 0, 4, 5, 6, 7, 8))
        self.assertEqual(apply_action(corner_state, "RIGHT"), (1, 0, 2, 3, 4, 5, 6, 7, 8))

    def test_get_successors(self):
        corner = (0, 1, 2, 3, 4, 5, 6, 7, 8)
        succ_corner = get_successors(corner)
        self.assertEqual(len(succ_corner), 2)
        actions = [act for act, _ in succ_corner]
        self.assertIn("DOWN", actions)
        self.assertIn("RIGHT", actions)

        edge = (1, 0, 2, 3, 4, 5, 6, 7, 8)
        self.assertEqual(len(get_successors(edge)), 3)

        center = (1, 2, 3, 4, 0, 5, 6, 7, 8)
        self.assertEqual(len(get_successors(center)), 4)

    def test_inversions_and_solvability(self):
        self.assertEqual(count_inversions(GOAL_STATE), 0)
        self.assertTrue(is_solvable(GOAL_STATE))

        unsolvable = (1, 2, 3, 4, 5, 6, 8, 7, 0)
        self.assertEqual(count_inversions(unsolvable), 1)
        self.assertFalse(is_solvable(unsolvable))

        solvable_2 = (1, 2, 3, 4, 6, 5, 8, 7, 0)
        self.assertEqual(count_inversions(solvable_2), 2)
        self.assertTrue(is_solvable(solvable_2))

    def test_parse_board(self):
        self.assertEqual(parse_board("1 2 3 4 5 6 7 8 0"), GOAL_STATE)
        self.assertEqual(parse_board("[1, 2, 3, 4, 5, 6, 7, 8, 0]"), GOAL_STATE)
        self.assertEqual(parse_board([[1, 2, 3], [4, 5, 6], [7, 8, 0]]), GOAL_STATE)

        with self.assertRaises(ValueError):
            parse_board("1 2 3 4 5")

        with self.assertRaises(ValueError):
            parse_board("1 2 3 4 5 6 7 8 9")

        with self.assertRaises(ValueError):
            parse_board("1 1 2 3 4 5 6 7 8")

    def test_format_board(self):
        formatted = format_board(GOAL_STATE)
        self.assertIn("1 │ 2 │ 3", formatted)
        self.assertIn("4 │ 5 │ 6", formatted)
        self.assertIn("7 │ 8 │  ", formatted)


if __name__ == "__main__":
    unittest.main()
