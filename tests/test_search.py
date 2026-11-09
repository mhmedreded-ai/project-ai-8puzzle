"""
test_search.py: Automated test suite for 8-Puzzle search algorithms using unittest.

Verifies:
  1. Already Solved Board (0 moves, immediate return)
  2. Short Path (1-3 steps across BFS, Greedy, and A*)
  3. Optimality Check (A* finds minimal path, Greedy can be suboptimal)
  4. Solvability Inversion Check (Immediate rejection of unsolvable boards)
  5. Dominance Check (h2 Manhattan expands <= nodes than h1 Misplaced)
"""

import unittest
from src.puzzle import GOAL_STATE, apply_action, is_solvable
from src.heuristics import misplaced_tiles, manhattan_distance
from src.search import a_star_search, greedy_search, bfs_search, SearchResult


class TestSearchAlgorithms(unittest.TestCase):
    """Unit tests for BFS, Greedy Best-First, and A* Search strategies."""

    def _assert_path_valid(self, initial_state, solution_path, expected_cost=None):
        """Helper to verify that solution path executes strictly valid transitions to the goal."""
        self.assertTrue(len(solution_path) > 0, "Solution path should not be empty")
        self.assertEqual(
            solution_path[0].state, initial_state, "Path must start at initial state"
        )
        self.assertEqual(
            solution_path[-1].state, GOAL_STATE, "Path must terminate at GOAL_STATE"
        )

        current = initial_state
        for node in solution_path[1:]:
            self.assertIsNotNone(node.action, "Intermediate node must record an action")
            current = apply_action(current, node.action)
            self.assertEqual(current, node.state, "Applied action must yield node state")

        self.assertEqual(current, GOAL_STATE)
        if expected_cost is not None:
            self.assertEqual(
                len(solution_path) - 1,
                expected_cost,
                f"Path length must equal expected cost {expected_cost}",
            )

    def test_1_already_solved_board(self):
        """Test 1: Already Solved Board: Verify algorithms immediately return 0 steps if given goal state."""
        goal = GOAL_STATE

        # 1. BFS
        bfs_res = bfs_search(goal)
        self.assertTrue(bfs_res.success)
        self.assertEqual(bfs_res.cost, 0)
        self.assertEqual(len(bfs_res.solution_path), 1)
        self.assertEqual(bfs_res.solution_path[0].state, goal)

        # 2. Greedy (Misplaced & Manhattan)
        greedy_h1 = greedy_search(goal, misplaced_tiles)
        self.assertTrue(greedy_h1.success)
        self.assertEqual(greedy_h1.cost, 0)
        self.assertEqual(len(greedy_h1.solution_path), 1)

        greedy_h2 = greedy_search(goal, manhattan_distance)
        self.assertTrue(greedy_h2.success)
        self.assertEqual(greedy_h2.cost, 0)
        self.assertEqual(len(greedy_h2.solution_path), 1)

        # 3. A* (Misplaced & Manhattan)
        astar_h1 = a_star_search(goal, misplaced_tiles)
        self.assertTrue(astar_h1.success)
        self.assertEqual(astar_h1.cost, 0)
        self.assertEqual(len(astar_h1.solution_path), 1)

        astar_h2 = a_star_search(goal, manhattan_distance)
        self.assertTrue(astar_h2.success)
        self.assertEqual(astar_h2.cost, 0)
        self.assertEqual(len(astar_h2.solution_path), 1)

    def test_2_short_path_1_to_3_steps(self):
        """Test 2: Short Path (1-3 steps): Verify all algorithms find the valid solution."""
        # 1-step board: Move blank RIGHT to reach goal
        # 1 2 3
        # 4 5 6
        # 7 0 8 -> blank at index 7 swaps with 8
        board_1_step = (1, 2, 3, 4, 5, 6, 7, 0, 8)

        # 2-step board: (1, 2, 3, 4, 5, 6, 0, 7, 8)
        board_2_step = (1, 2, 3, 4, 5, 6, 0, 7, 8)

        # 3-step board: (1, 2, 3, 0, 4, 6, 7, 5, 8)
        board_3_step = (1, 2, 3, 0, 4, 6, 7, 5, 8)

        test_cases = [
            (board_1_step, 1),
            (board_2_step, 2),
            (board_3_step, 3),
        ]

        for board, expected_depth in test_cases:
            # BFS
            res_bfs = bfs_search(board)
            self.assertTrue(res_bfs.success)
            self.assertEqual(res_bfs.cost, expected_depth)
            self._assert_path_valid(board, res_bfs.solution_path, expected_depth)

            # Greedy Misplaced
            res_g_h1 = greedy_search(board, misplaced_tiles)
            self.assertTrue(res_g_h1.success)
            self._assert_path_valid(board, res_g_h1.solution_path)

            # Greedy Manhattan
            res_g_h2 = greedy_search(board, manhattan_distance)
            self.assertTrue(res_g_h2.success)
            self._assert_path_valid(board, res_g_h2.solution_path)

            # A* Misplaced
            res_a_h1 = a_star_search(board, misplaced_tiles)
            self.assertTrue(res_a_h1.success)
            self.assertEqual(res_a_h1.cost, expected_depth)
            self._assert_path_valid(board, res_a_h1.solution_path, expected_depth)

            # A* Manhattan
            res_a_h2 = a_star_search(board, manhattan_distance)
            self.assertTrue(res_a_h2.success)
            self.assertEqual(res_a_h2.cost, expected_depth)
            self._assert_path_valid(board, res_a_h2.solution_path, expected_depth)

    def test_3_optimality_check(self):
        """Test 3: Optimality Check: Verify A* finds optimal minimal steps, while Greedy can find a longer path."""
        # Medium board with optimal depth 12:
        # Greedy with Misplaced heuristic is lured into local detours, taking 42 moves!
        board_d12 = (5, 8, 2, 1, 7, 3, 0, 4, 6)

        # A* Search must strictly find the optimal minimal cost (12)
        astar_h1 = a_star_search(board_d12, misplaced_tiles)
        astar_h2 = a_star_search(board_d12, manhattan_distance)

        self.assertTrue(astar_h1.success)
        self.assertTrue(astar_h2.success)
        self.assertEqual(astar_h1.cost, 12, "A*(h1) must return optimal path length 12")
        self.assertEqual(astar_h2.cost, 12, "A*(h2) must return optimal path length 12")
        self._assert_path_valid(board_d12, astar_h1.solution_path, 12)
        self._assert_path_valid(board_d12, astar_h2.solution_path, 12)

        # Greedy Search with Misplaced Tiles is suboptimal
        greedy_h1 = greedy_search(board_d12, misplaced_tiles)
        self.assertTrue(greedy_h1.success)
        self.assertGreater(
            greedy_h1.cost,
            astar_h2.cost,
            f"Greedy cost ({greedy_h1.cost}) should be greater than optimal A* cost ({astar_h2.cost})",
        )
        self._assert_path_valid(board_d12, greedy_h1.solution_path, greedy_h1.cost)

    def test_4_solvability_inversion_check(self):
        """Test 4: Solvability Inversion Check: Verify unsolvable boards are caught immediately without looping."""
        # Unsolvable board: swap tiles 1 and 2 on the goal board
        # (2, 1, 3, 4, 5, 6, 7, 8, 0) -> exactly 1 inversion (odd parity)
        unsolvable_board = (2, 1, 3, 4, 5, 6, 7, 8, 0)

        # Solvability check must reject
        self.assertFalse(
            is_solvable(unsolvable_board),
            "Board with odd parity must be detected as unsolvable",
        )

        # All search routines must abort immediately without infinite loop or node expansions
        bfs_res = bfs_search(unsolvable_board)
        self.assertFalse(bfs_res.success)
        self.assertEqual(bfs_res.nodes_expanded, 0)
        self.assertEqual(len(bfs_res.solution_path), 0)

        greedy_res = greedy_search(unsolvable_board, manhattan_distance)
        self.assertFalse(greedy_res.success)
        self.assertEqual(greedy_res.nodes_expanded, 0)

        astar_res = a_star_search(unsolvable_board, manhattan_distance)
        self.assertFalse(astar_res.success)
        self.assertEqual(astar_res.nodes_expanded, 0)

    def test_5_dominance_check(self):
        """Test 5: Dominance Check: Assert that h2 (Manhattan) expands <= nodes than h1 (Misplaced)."""
        # Test on depth-12 board
        board_d12 = (5, 8, 2, 1, 7, 3, 0, 4, 6)
        astar_h1 = a_star_search(board_d12, misplaced_tiles)
        astar_h2 = a_star_search(board_d12, manhattan_distance)

        self.assertTrue(astar_h1.success)
        self.assertTrue(astar_h2.success)
        self.assertLessEqual(
            astar_h2.nodes_expanded,
            astar_h1.nodes_expanded,
            f"A*(h2) expanded ({astar_h2.nodes_expanded}) must be <= A*(h1) ({astar_h1.nodes_expanded})",
        )

        # Test on depth-20 board (dramatically demonstrates dominance)
        board_d20 = (8, 7, 0, 5, 4, 2, 1, 6, 3)
        astar_h1_deep = a_star_search(board_d20, misplaced_tiles)
        astar_h2_deep = a_star_search(board_d20, manhattan_distance)

        self.assertTrue(astar_h1_deep.success)
        self.assertTrue(astar_h2_deep.success)
        self.assertLess(
            astar_h2_deep.nodes_expanded,
            astar_h1_deep.nodes_expanded,
            f"A*(h2) must expand strictly fewer nodes than A*(h1) on deep search",
        )

    def test_tuple_unpacking(self):
        """Test tuple unpacking interface: path, nodes, time = search_fn(...)."""
        board = (1, 2, 3, 4, 5, 6, 7, 0, 8)
        path, nodes, elapsed = a_star_search(board, manhattan_distance)
        self.assertIsInstance(path, list)
        self.assertIsInstance(nodes, int)
        self.assertIsInstance(elapsed, float)
        self.assertEqual(len(path), 2)
        self.assertGreaterEqual(nodes, 1)


if __name__ == "__main__":
    unittest.main()
