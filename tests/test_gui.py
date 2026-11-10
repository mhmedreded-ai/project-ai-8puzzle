"""
test_gui.py: Automated unit tests for the CustomTkinter 8-Puzzle GUI (gui.py).

Verifies:
  1. Preset boards and solvability parity invariant
  2. Random solvable board generation
  3. Algorithm mapping & search integration
  4. GUI playback state transitions (Step Forward, Step Backward, Reset, Speed Slider)
"""

import unittest
import gui
from src.puzzle import GOAL_STATE, is_solvable


class TestEightPuzzleGUI(unittest.TestCase):
    """Tests for gui.py components and controller logic."""

    @classmethod
    def setUpClass(cls):
        cls.app = gui.EightPuzzleGUI()
        cls.app.withdraw()  # Hide window during automated test execution

    @classmethod
    def tearDownClass(cls):
        cls.app.destroy()

    def test_1_presets_and_solvability_badge(self):
        """Verify all presets load properly and update the solvability badge."""
        for name, expected_state in gui.PRESET_BOARDS.items():
            self.app._on_preset_selected(name)
            self.assertEqual(self.app.initial_state, expected_state)
            self.assertEqual(self.app.current_state, expected_state)
            badge_text = self.app.solvability_badge.cget("text")
            if is_solvable(expected_state):
                self.assertIn("SOLVABLE", badge_text)
                self.assertNotIn("UNSOLVABLE", badge_text)
            else:
                self.assertIn("UNSOLVABLE", badge_text)

    def test_2_random_solvable_generator(self):
        """Verify random board generator always yields a valid solvable permutation."""
        for _ in range(15):
            self.app._on_generate_random_solvable()
            state = self.app.initial_state
            self.assertEqual(sorted(state), list(range(9)))
            self.assertTrue(is_solvable(state, GOAL_STATE))
            self.assertNotEqual(state, GOAL_STATE)

    def test_3_custom_board_entry_and_validation(self):
        """Verify custom 9-integer entry applies valid states and catches invalid input."""
        self.app.custom_entry.delete(0, "end")
        self.app.custom_entry.insert(0, "1 2 3 4 5 6 7 0 8")
        self.app._on_apply_custom_board()
        self.assertEqual(self.app.initial_state, (1, 2, 3, 4, 5, 6, 7, 0, 8))

        # Invalid board input should not overwrite initial_state
        self.app.custom_entry.delete(0, "end")
        self.app.custom_entry.insert(0, "invalid board")
        self.app._on_apply_custom_board()
        self.assertEqual(self.app.initial_state, (1, 2, 3, 4, 5, 6, 7, 0, 8))
        self.assertIn("Invalid", self.app.solvability_badge.cget("text"))

    def test_4_search_and_playback_controls(self):
        """Verify solving a preset updates metrics and enables step-by-step playback."""
        self.app._on_preset_selected("Easy (d=4)")
        algo_name = "A* Search (Manhattan - h2)"
        result = gui.ALGORITHMS[algo_name](self.app.initial_state)
        self.app._on_search_complete(algo_name, result)

        self.assertEqual(self.app.card_cost.value_label.cget("text"), "4")
        self.assertEqual(len(self.app.solution_path), 5)
        self.assertEqual(self.app.current_step_index, 0)

        # Step forward twice
        self.app._on_step_forward()
        self.assertEqual(self.app.current_step_index, 1)
        self.app._on_step_forward()
        self.assertEqual(self.app.current_step_index, 2)

        # Step backward once
        self.app._on_step_backward()
        self.assertEqual(self.app.current_step_index, 1)

        # Reset to initial
        self.app._on_reset_to_initial()
        self.assertEqual(self.app.current_step_index, 0)
        self.assertEqual(self.app.current_state, gui.PRESET_BOARDS["Easy (d=4)"])

        # Speed slider update
        self.app._on_speed_changed(0.15)
        self.assertAlmostEqual(self.app.step_delay_seconds, 0.15)


if __name__ == "__main__":
    unittest.main()
