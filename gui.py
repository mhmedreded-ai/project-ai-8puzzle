"""
gui.py: Modern Graphical User Interface for the 8-Puzzle Solver & Benchmark Suite.

Built with CustomTkinter (dark mode), providing:
  1. Interactive 3x3 Visual Puzzle Board with dynamic state transitions
  2. Board Setup Controls (Presets, Custom 9-integer Entry, Random Solvable Generator,
     and Real-Time Solvability Parity Badge)
  3. Algorithm & Heuristic Selector with non-blocking threaded search execution
  4. Playback & Animation Controller (Play, Pause, Step Forward/Backward, Reset, Speed Slider)
  5. Performance Metrics Dashboard Cards (Path Cost, Nodes Expanded, Nodes Generated,
     Max Frontier Size, Execution Time)
"""

from __future__ import annotations

import random
import threading
from typing import Callable, Optional

import customtkinter as ctk

from src.heuristics import manhattan_distance, misplaced_tiles
from src.puzzle import (
    GOAL_STATE,
    Node,
    State,
    count_inversions,
    is_solvable,
    parse_board,
)
from src.search import (
    SearchResult,
    a_star_search,
    bfs_search,
    greedy_search as greedy_best_first_search,
)

# Configure global CustomTkinter appearance
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


# =====================================================================
# Constants & Color Palette
# =====================================================================

PRESET_BOARDS: dict[str, State] = {
    "Easy (d=4)": (1, 5, 2, 4, 0, 3, 7, 8, 6),
    "Medium (d=12)": (5, 8, 2, 1, 7, 3, 0, 4, 6),
    "Hard (d=20)": (8, 7, 0, 5, 4, 2, 1, 6, 3),
    "Unsolvable": (1, 2, 3, 4, 5, 6, 8, 7, 0),
}

ALGORITHMS: dict[str, Callable[[State], SearchResult]] = {
    "Breadth-First Search (BFS)": lambda s: bfs_search(s, GOAL_STATE),
    "Greedy Best-First Search (Misplaced - h1)": lambda s: greedy_best_first_search(
        s, misplaced_tiles, GOAL_STATE
    ),
    "Greedy Best-First Search (Manhattan - h2)": lambda s: greedy_best_first_search(
        s, manhattan_distance, GOAL_STATE
    ),
    "A* Search (Misplaced - h1)": lambda s: a_star_search(
        s, misplaced_tiles, GOAL_STATE
    ),
    "A* Search (Manhattan - h2)": lambda s: a_star_search(
        s, manhattan_distance, GOAL_STATE
    ),
}

COLORS = {
    "bg_card": "#1E293B",
    "bg_board": "#0F172A",
    "tile_default": "#334155",
    "tile_goal": "#1D4ED8",
    "tile_blank": "#090D16",
    "tile_border_blank": "#1E293B",
    "tile_border_active": "#3B82F6",
    "text_primary": "#F8FAFC",
    "text_muted": "#94A3B8",
    "badge_solvable_bg": "#065F46",
    "badge_solvable_fg": "#34D399",
    "badge_unsolvable_bg": "#7F1D1D",
    "badge_unsolvable_fg": "#F87171",
    "badge_invalid_bg": "#78350F",
    "badge_invalid_fg": "#FBBF24",
    "accent_green": "#10B981",
    "accent_blue": "#3B82F6",
    "accent_amber": "#F59E0B",
}


# =====================================================================
# Reusable UI Components
# =====================================================================


class MetricCard(ctk.CTkFrame):
    """
    Dashboard card widget displaying a single performance metric title and value.
    """

    def __init__(self, master: ctk.CTkBaseClass, title: str, initial_value: str = "—") -> None:
        super().__init__(master, fg_color=COLORS["bg_card"], corner_radius=10)
        self.grid_columnconfigure(0, weight=1)

        self.title_label = ctk.CTkLabel(
            self,
            text=title.upper(),
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=COLORS["text_muted"],
        )
        self.title_label.grid(row=0, column=0, padx=12, pady=(10, 2), sticky="w")

        self.value_label = ctk.CTkLabel(
            self,
            text=initial_value,
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=COLORS["text_primary"],
        )
        self.value_label.grid(row=1, column=0, padx=12, pady=(0, 10), sticky="w")

    def set_value(self, value: str, color: Optional[str] = None) -> None:
        """Updates the displayed metric text and optional foreground color."""
        self.value_label.configure(
            text=value,
            text_color=color if color is not None else COLORS["text_primary"],
        )


class PuzzleBoardWidget(ctk.CTkFrame):
    """
    Visual 3x3 grid representation of the 8-Puzzle board.
    Renders tiles 1-8 with distinct styling for the blank tile (0) and tiles in goal position.
    """

    def __init__(
        self,
        master: ctk.CTkBaseClass,
        initial_state: State = GOAL_STATE,
        on_tile_click: Optional[Callable[[int], None]] = None,
    ) -> None:
        super().__init__(master, fg_color=COLORS["bg_board"], corner_radius=14)
        self.on_tile_click = on_tile_click
        self.tiles: list[ctk.CTkButton] = []

        for idx in range(3):
            self.grid_rowconfigure(idx, weight=1, uniform="cell")
            self.grid_columnconfigure(idx, weight=1, uniform="cell")

        self._build_tiles()
        self.update_state(initial_state)

    def _build_tiles(self) -> None:
        """Creates the 9 tile buttons inside the 3x3 grid."""
        tile_font = ctk.CTkFont(family="Segoe UI", size=34, weight="bold")
        for idx in range(9):
            row, col = divmod(idx, 3)
            btn = ctk.CTkButton(
                self,
                text="",
                font=tile_font,
                width=105,
                height=105,
                corner_radius=12,
                border_width=2,
                command=lambda i=idx: self._handle_click(i),
            )
            btn.grid(row=row, column=col, padx=7, pady=7, sticky="nsew")
            self.tiles.append(btn)

    def _handle_click(self, index: int) -> None:
        if self.on_tile_click is not None:
            self.on_tile_click(index)

    def update_state(self, state: State) -> None:
        """
        Updates the 3x3 visual tiles to reflect the given 9-tuple board state.
        """
        for idx, val in enumerate(state):
            btn = self.tiles[idx]
            if val == 0:
                # Blank tile (0)
                btn.configure(
                    text="",
                    fg_color=COLORS["tile_blank"],
                    hover_color=COLORS["tile_blank"],
                    border_color=COLORS["tile_border_blank"],
                    text_color=COLORS["text_muted"],
                    state="disabled",
                )
            else:
                in_goal_spot = val == GOAL_STATE[idx]
                tile_color = COLORS["tile_goal"] if in_goal_spot else COLORS["tile_default"]
                border_color = (
                    COLORS["tile_border_active"] if in_goal_spot else "#475569"
                )
                btn.configure(
                    text=str(val),
                    fg_color=tile_color,
                    hover_color="#2563EB" if in_goal_spot else "#475569",
                    border_color=border_color,
                    text_color=COLORS["text_primary"],
                    state="normal",
                )


# =====================================================================
# Main Application Window
# =====================================================================


class EightPuzzleGUI(ctk.CTk):
    """
    Main CustomTkinter application window for the 8-Puzzle AI Solver.
    Integrates board configuration, search execution, animation playback,
    and empirical performance metrics.
    """

    def __init__(self) -> None:
        super().__init__()

        self.title("8-Puzzle AI Solver & Heuristic Search Benchmark")
        self.geometry("1120x760")
        self.minsize(980, 680)

        # Application state variables
        self.initial_state: State = PRESET_BOARDS["Easy (d=4)"]
        self.current_state: State = self.initial_state
        self.solution_path: list[Node] = []
        self.current_step_index: int = 0

        # Animation & Concurrency state
        self.is_playing: bool = False
        self.animation_after_id: Optional[str] = None
        self.is_searching: bool = False
        self.step_delay_seconds: float = 0.35

        # Configure top-level grid layout (Sidebar on Left, Main Workspace on Right)
        self.grid_columnconfigure(0, weight=0, minsize=380)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_main_workspace()

        # Load initial preset and validate
        self._apply_board_state(self.initial_state, update_entry=True)

    # -----------------------------------------------------------------
    # UI Construction
    # -----------------------------------------------------------------

    def _build_sidebar(self) -> None:
        """Builds the left control panel containing board setup and algorithm controls."""
        self.sidebar = ctk.CTkFrame(self, width=380, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=(0, 2), pady=0)
        self.sidebar.grid_columnconfigure(0, weight=1)

        # App Header
        header_label = ctk.CTkLabel(
            self.sidebar,
            text="8-Puzzle AI Solver",
            font=ctk.CTkFont(size=22, weight="bold"),
        )
        header_label.grid(row=0, column=0, padx=20, pady=(20, 2), sticky="w")

        sub_label = ctk.CTkLabel(
            self.sidebar,
            text="State-Space & Heuristic Search Suite",
            font=ctk.CTkFont(size=12),
            text_color=COLORS["text_muted"],
        )
        sub_label.grid(row=1, column=0, padx=20, pady=(0, 16), sticky="w")

        # --- Section 1: Board Setup & Input Controls ---
        setup_frame = ctk.CTkFrame(self.sidebar, fg_color=COLORS["bg_card"], corner_radius=12)
        setup_frame.grid(row=2, column=0, padx=16, pady=(0, 14), sticky="ew")
        setup_frame.grid_columnconfigure(0, weight=1)

        sec1_title = ctk.CTkLabel(
            setup_frame,
            text="1. BOARD CONFIGURATION",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS["accent_blue"],
        )
        sec1_title.grid(row=0, column=0, columnspan=2, padx=14, pady=(12, 8), sticky="w")

        # Preset Dropdown
        preset_lbl = ctk.CTkLabel(
            setup_frame, text="Select Curated Preset:", font=ctk.CTkFont(size=12)
        )
        preset_lbl.grid(row=1, column=0, columnspan=2, padx=14, pady=(0, 4), sticky="w")

        self.preset_var = ctk.StringVar(value="Easy (d=4)")
        self.preset_menu = ctk.CTkOptionMenu(
            setup_frame,
            values=list(PRESET_BOARDS.keys()),
            variable=self.preset_var,
            command=self._on_preset_selected,
            height=34,
        )
        self.preset_menu.grid(row=2, column=0, columnspan=2, padx=14, pady=(0, 10), sticky="ew")

        # Custom 9-Integer Board Input
        custom_lbl = ctk.CTkLabel(
            setup_frame,
            text="Custom Board (9 digits 0-8, 0 = blank):",
            font=ctk.CTkFont(size=12),
        )
        custom_lbl.grid(row=3, column=0, columnspan=2, padx=14, pady=(0, 4), sticky="w")

        self.custom_entry = ctk.CTkEntry(
            setup_frame,
            placeholder_text="e.g. 1 5 2 4 0 3 7 8 6",
            height=34,
        )
        self.custom_entry.grid(row=4, column=0, padx=(14, 6), pady=(0, 10), sticky="ew")
        self.custom_entry.bind("<KeyRelease>", self._on_custom_entry_changed)
        self.custom_entry.bind("<Return>", lambda _e: self._on_apply_custom_board())

        self.apply_custom_btn = ctk.CTkButton(
            setup_frame,
            text="Apply",
            width=72,
            height=34,
            command=self._on_apply_custom_board,
        )
        self.apply_custom_btn.grid(row=4, column=1, padx=(0, 14), pady=(0, 10))

        # Random Solvable Board Generator Button
        self.random_btn = ctk.CTkButton(
            setup_frame,
            text="🎲  Generate Random Solvable Board",
            fg_color="#334155",
            hover_color="#475569",
            height=36,
            command=self._on_generate_random_solvable,
        )
        self.random_btn.grid(row=5, column=0, columnspan=2, padx=14, pady=(0, 12), sticky="ew")

        # Real-Time Solvability Validator Badge
        self.solvability_badge = ctk.CTkLabel(
            setup_frame,
            text="✓ SOLVABLE (0 Inversions - Even Parity)",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=COLORS["badge_solvable_bg"],
            text_color=COLORS["badge_solvable_fg"],
            corner_radius=8,
            height=34,
        )
        self.solvability_badge.grid(
            row=6, column=0, columnspan=2, padx=14, pady=(0, 14), sticky="ew"
        )

        # --- Section 2: Algorithm & Heuristic Selector ---
        algo_frame = ctk.CTkFrame(self.sidebar, fg_color=COLORS["bg_card"], corner_radius=12)
        algo_frame.grid(row=3, column=0, padx=16, pady=(0, 14), sticky="ew")
        algo_frame.grid_columnconfigure(0, weight=1)

        sec2_title = ctk.CTkLabel(
            algo_frame,
            text="2. SEARCH ALGORITHM & HEURISTIC",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLORS["accent_blue"],
        )
        sec2_title.grid(row=0, column=0, padx=14, pady=(12, 8), sticky="w")

        self.algo_var = ctk.StringVar(value="A* Search (Manhattan - h2)")

        for idx, algo_label in enumerate(ALGORITHMS.keys(), start=1):
            rb = ctk.CTkRadioButton(
                algo_frame,
                text=algo_label,
                variable=self.algo_var,
                value=algo_label,
                font=ctk.CTkFont(size=12),
            )
            rb.grid(row=idx, column=0, padx=16, pady=5, sticky="w")

        # Solve Button
        self.solve_btn = ctk.CTkButton(
            algo_frame,
            text="▶  Solve Puzzle",
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color=COLORS["accent_green"],
            hover_color="#059669",
            height=42,
            command=self._on_solve_clicked,
        )
        self.solve_btn.grid(row=len(ALGORITHMS) + 1, column=0, padx=14, pady=(12, 14), sticky="ew")

        # --- Status Message Footer ---
        self.status_label = ctk.CTkLabel(
            self.sidebar,
            text="Ready. Select a board and click Solve.",
            font=ctk.CTkFont(size=12),
            text_color=COLORS["text_muted"],
            wraplength=340,
            justify="left",
        )
        self.status_label.grid(row=4, column=0, padx=20, pady=(4, 16), sticky="w")

    def _build_main_workspace(self) -> None:
        """Builds the right workspace containing metrics cards, the 3x3 board, and playback controls."""
        self.workspace = ctk.CTkFrame(self, fg_color="transparent")
        self.workspace.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.workspace.grid_columnconfigure(0, weight=1)
        self.workspace.grid_rowconfigure(1, weight=1)

        # --- Top Row: Performance Metrics Dashboard ---
        metrics_container = ctk.CTkFrame(self.workspace, fg_color="transparent")
        metrics_container.grid(row=0, column=0, sticky="ew", pady=(0, 16))
        for i in range(5):
            metrics_container.grid_columnconfigure(i, weight=1, uniform="metric")

        self.card_cost = MetricCard(metrics_container, "Path Cost (d)")
        self.card_cost.grid(row=0, column=0, padx=(0, 6), sticky="ew")

        self.card_expanded = MetricCard(metrics_container, "Nodes Expanded")
        self.card_expanded.grid(row=0, column=1, padx=6, sticky="ew")

        self.card_generated = MetricCard(metrics_container, "Nodes Generated")
        self.card_generated.grid(row=0, column=2, padx=6, sticky="ew")

        self.card_frontier = MetricCard(metrics_container, "Max Frontier")
        self.card_frontier.grid(row=0, column=3, padx=6, sticky="ew")

        self.card_time = MetricCard(metrics_container, "Exec Time (ms)")
        self.card_time.grid(row=0, column=4, padx=(6, 0), sticky="ew")

        # --- Middle Row: Visual 3x3 Board & Node Telemetry ---
        center_frame = ctk.CTkFrame(self.workspace, fg_color=COLORS["bg_card"], corner_radius=14)
        center_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 16))
        center_frame.grid_columnconfigure(0, weight=1)
        center_frame.grid_rowconfigure(1, weight=1)

        # Telemetry Header Bar inside Board Area
        telemetry_bar = ctk.CTkFrame(center_frame, fg_color="transparent")
        telemetry_bar.grid(row=0, column=0, sticky="ew", padx=20, pady=(14, 4))
        telemetry_bar.grid_columnconfigure(0, weight=1)
        telemetry_bar.grid_columnconfigure(1, weight=1)

        self.step_counter_label = ctk.CTkLabel(
            telemetry_bar,
            text="Step 0 / 0",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        self.step_counter_label.grid(row=0, column=0, sticky="w")

        self.node_info_label = ctk.CTkLabel(
            telemetry_bar,
            text="Action: Initial  |  g(n) = 0  |  h(n) = —  |  f(n) = —",
            font=ctk.CTkFont(size=13),
            text_color=COLORS["text_muted"],
        )
        self.node_info_label.grid(row=0, column=1, sticky="e")

        # 3x3 Visual Puzzle Grid
        board_holder = ctk.CTkFrame(center_frame, fg_color="transparent")
        board_holder.grid(row=1, column=0, padx=20, pady=10)

        self.board_widget = PuzzleBoardWidget(
            board_holder,
            initial_state=self.current_state,
            on_tile_click=self._on_board_tile_clicked,
        )
        self.board_widget.pack(expand=True)

        legend_label = ctk.CTkLabel(
            center_frame,
            text="Tip: Blue tiles are in their canonical goal positions (1–8). Click adjacent tiles to slide manually.",
            font=ctk.CTkFont(size=11),
            text_color=COLORS["text_muted"],
        )
        legend_label.grid(row=2, column=0, pady=(0, 12))

        # --- Bottom Row: Playback & Animation Controls ---
        playback_frame = ctk.CTkFrame(self.workspace, fg_color=COLORS["bg_card"], corner_radius=12)
        playback_frame.grid(row=2, column=0, sticky="ew")
        playback_frame.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)

        btn_Container = ctk.CTkFrame(playback_frame, fg_color="transparent")
        btn_Container.grid(row=0, column=0, columnspan=5, padx=16, pady=(14, 8), sticky="ew")
        for i in range(5):
            btn_Container.grid_columnconfigure(i, weight=1)

        self.btn_reset = ctk.CTkButton(
            btn_Container,
            text="⏮  Reset to Initial",
            fg_color="#334155",
            hover_color="#475569",
            height=36,
            command=self._on_reset_to_initial,
            state="disabled",
        )
        self.btn_reset.grid(row=0, column=0, padx=4, sticky="ew")

        self.btn_step_back = ctk.CTkButton(
            btn_Container,
            text="⏪  Step Backward",
            fg_color="#334155",
            hover_color="#475569",
            height=36,
            command=self._on_step_backward,
            state="disabled",
        )
        self.btn_step_back.grid(row=0, column=1, padx=4, sticky="ew")

        self.btn_play = ctk.CTkButton(
            btn_Container,
            text="▶  Play / Animate",
            fg_color=COLORS["accent_blue"],
            hover_color="#2563EB",
            height=36,
            command=self._on_play_clicked,
            state="disabled",
        )
        self.btn_play.grid(row=0, column=2, padx=4, sticky="ew")

        self.btn_pause = ctk.CTkButton(
            btn_Container,
            text="⏸  Pause",
            fg_color=COLORS["accent_amber"],
            hover_color="#D97706",
            height=36,
            command=self._on_pause_clicked,
            state="disabled",
        )
        self.btn_pause.grid(row=0, column=3, padx=4, sticky="ew")

        self.btn_step_fwd = ctk.CTkButton(
            btn_Container,
            text="Step Forward  ⏩",
            fg_color="#334155",
            hover_color="#475569",
            height=36,
            command=self._on_step_forward,
            state="disabled",
        )
        self.btn_step_fwd.grid(row=0, column=4, padx=4, sticky="ew")

        # Speed Slider Row (0.05s to 1.0s)
        slider_row = ctk.CTkFrame(playback_frame, fg_color="transparent")
        slider_row.grid(row=1, column=0, columnspan=5, padx=20, pady=(2, 14), sticky="ew")
        slider_row.grid_columnconfigure(1, weight=1)

        speed_title = ctk.CTkLabel(
            slider_row,
            text="Animation Delay per Step:",
            font=ctk.CTkFont(size=12),
            text_color=COLORS["text_muted"],
        )
        speed_title.grid(row=0, column=0, padx=(0, 12), sticky="w")

        self.speed_slider = ctk.CTkSlider(
            slider_row,
            from_=0.05,
            to=1.00,
            number_of_steps=19,
            command=self._on_speed_changed,
        )
        self.speed_slider.set(self.step_delay_seconds)
        self.speed_slider.grid(row=0, column=1, padx=8, sticky="ew")

        self.speed_value_label = ctk.CTkLabel(
            slider_row,
            text=f"{self.step_delay_seconds:.2f} s / step",
            font=ctk.CTkFont(size=12, weight="bold"),
            width=95,
        )
        self.speed_value_label.grid(row=0, column=2, padx=(8, 0), sticky="e")

    # -----------------------------------------------------------------
    # Board Input & Validation Logic
    # -----------------------------------------------------------------

    def _on_preset_selected(self, preset_name: str) -> None:
        """Handles selection from the curated preset dropdown."""
        if self.is_searching:
            return
        state = PRESET_BOARDS[preset_name]
        self._apply_board_state(state, update_entry=True)
        self.status_label.configure(
            text=f"Loaded preset '{preset_name}'. Ready to solve.",
            text_color=COLORS["text_muted"],
        )

    def _on_custom_entry_changed(self, _event: object = None) -> None:
        """Real-time solvability validation as the user types in the custom board box."""
        raw_text = self.custom_entry.get().strip()
        if not raw_text:
            return
        try:
            state = parse_board(raw_text)
            self._update_solvability_badge(state)
        except ValueError as exc:
            self.solvability_badge.configure(
                text=f"⚠ Invalid Input: {exc}",
                fg_color=COLORS["badge_invalid_bg"],
                text_color=COLORS["badge_invalid_fg"],
            )

    def _on_apply_custom_board(self) -> None:
        """Parses and applies the custom 9-integer board entered by the user."""
        if self.is_searching:
            return
        raw_text = self.custom_entry.get().strip()
        try:
            state = parse_board(raw_text)
        except ValueError as exc:
            self.solvability_badge.configure(
                text=f"⚠ Invalid Board: {exc}",
                fg_color=COLORS["badge_invalid_bg"],
                text_color=COLORS["badge_invalid_fg"],
            )
            self.status_label.configure(
                text=f"Custom board error: {exc}",
                text_color=COLORS["badge_unsolvable_fg"],
            )
            return

        self._apply_board_state(state, update_entry=True)
        self.status_label.configure(
            text=f"Applied custom board {state}.",
            text_color=COLORS["text_muted"],
        )

    def _on_generate_random_solvable(self) -> None:
        """
        Generates a uniformly random permutation of {0..8} that satisfies
        the even-inversion parity invariant (guaranteed solvable).
        """
        if self.is_searching:
            return

        tiles = list(range(9))
        while True:
            random.shuffle(tiles)
            candidate: State = tuple(tiles)
            if candidate != GOAL_STATE and is_solvable(candidate, GOAL_STATE):
                break

        self._apply_board_state(candidate, update_entry=True)
        self.status_label.configure(
            text=f"Generated random solvable board ({count_inversions(candidate)} inversions).",
            text_color=COLORS["text_muted"],
        )

    def _on_board_tile_clicked(self, clicked_idx: int) -> None:
        """Allows interactive manual sliding when clicking a tile adjacent to the blank."""
        if self.is_searching or self.is_playing:
            return

        blank_idx = self.current_state.index(0)
        r1, c1 = divmod(clicked_idx, 3)
        r2, c2 = divmod(blank_idx, 3)

        if abs(r1 - r2) + abs(c1 - c2) == 1:
            new_list = list(self.current_state)
            new_list[blank_idx], new_list[clicked_idx] = (
                new_list[clicked_idx],
                new_list[blank_idx],
            )
            new_state = tuple(new_list)
            self._apply_board_state(new_state, update_entry=True)
            self.status_label.configure(
                text="Board updated manually. Click Solve to compute optimal path.",
                text_color=COLORS["text_muted"],
            )

    def _apply_board_state(self, state: State, update_entry: bool = True) -> None:
        """Sets a new initial board state, resets any active playback, and updates visuals."""
        self._stop_animation()
        self.initial_state = state
        self.current_state = state
        self.solution_path = []
        self.current_step_index = 0

        if update_entry:
            self.custom_entry.delete(0, "end")
            self.custom_entry.insert(0, " ".join(str(x) for x in state))

        self.board_widget.update_state(state)
        self._update_solvability_badge(state)
        self._reset_metrics_and_telemetry(state)
        self._update_playback_buttons()

    def _update_solvability_badge(self, state: State) -> None:
        """Updates the real-time solvability indicator badge and solve button state."""
        inv = count_inversions(state)
        solvable = is_solvable(state, GOAL_STATE)
        parity = "Even" if inv % 2 == 0 else "Odd"

        if solvable:
            self.solvability_badge.configure(
                text=f"✓ SOLVABLE  ({inv} Inversions — {parity} Parity)",
                fg_color=COLORS["badge_solvable_bg"],
                text_color=COLORS["badge_solvable_fg"],
            )
            if not self.is_searching:
                self.solve_btn.configure(state="normal")
        else:
            self.solvability_badge.configure(
                text=f"✗ UNSOLVABLE  ({inv} Inversions — {parity} Parity)",
                fg_color=COLORS["badge_unsolvable_bg"],
                text_color=COLORS["badge_unsolvable_fg"],
            )

    def _reset_metrics_and_telemetry(self, state: State) -> None:
        """Resets metric dashboard cards and node telemetry for a fresh board."""
        self.card_cost.set_value("—")
        self.card_expanded.set_value("—")
        self.card_generated.set_value("—")
        self.card_frontier.set_value("—")
        self.card_time.set_value("—")

        h1 = misplaced_tiles(state, GOAL_STATE)
        h2 = manhattan_distance(state, GOAL_STATE)
        self.step_counter_label.configure(text="Step 0 / 0")
        self.node_info_label.configure(
            text=f"Initial State  |  h1 (Misplaced) = {h1}  |  h2 (Manhattan) = {h2}"
        )

    # -----------------------------------------------------------------
    # Threaded Search Execution
    # -----------------------------------------------------------------

    def _on_solve_clicked(self) -> None:
        """Starts the selected search algorithm on a background daemon thread."""
        if self.is_searching:
            return

        self._stop_animation()

        # Always reset to the configured initial board before solving
        state_to_solve = self.initial_state
        algo_name = self.algo_var.get()
        search_callable = ALGORITHMS[algo_name]

        self.is_searching = True
        self.solve_btn.configure(state="disabled", text="⏳  Searching...")
        self._disable_all_playback_buttons()
        self.status_label.configure(
            text=f"Running {algo_name} in background thread...",
            text_color=COLORS["accent_amber"],
        )

        worker = threading.Thread(
            target=self._run_search_worker,
            args=(state_to_solve, algo_name, search_callable),
            daemon=True,
        )
        worker.start()

    def _run_search_worker(
        self,
        state: State,
        algo_name: str,
        search_callable: Callable[[State], SearchResult],
    ) -> None:
        """Background thread routine executing the search algorithm."""
        try:
            result = search_callable(state)
            self.after(0, lambda: self._on_search_complete(algo_name, result))
        except Exception as exc:
            self.after(0, lambda: self._on_search_error(str(exc)))

    def _on_search_complete(self, algo_name: str, result: SearchResult) -> None:
        """Main-thread callback invoked when the background search finishes."""
        self.is_searching = False
        self.solve_btn.configure(state="normal", text="▶  Solve Puzzle")

        if not result.success:
            self.solution_path = []
            self.current_step_index = 0
            self.card_cost.set_value("Unsolvable", color=COLORS["badge_unsolvable_fg"])
            self.card_expanded.set_value(f"{result.nodes_expanded:,}")
            self.card_generated.set_value(f"{result.nodes_generated:,}")
            self.card_frontier.set_value(f"{result.max_frontier_size:,}")
            self.card_time.set_value(f"{result.elapsed_time_ms:.2f}")
            self.status_label.configure(
                text=(
                    f"Search aborted: Board is mathematically unsolvable "
                    f"({count_inversions(self.initial_state)} inversions, odd parity)."
                ),
                text_color=COLORS["badge_unsolvable_fg"],
            )
            self._update_playback_buttons()
            return

        # Populate Performance Metrics Dashboard Cards
        self.card_cost.set_value(str(result.cost), color=COLORS["badge_solvable_fg"])
        self.card_expanded.set_value(f"{result.nodes_expanded:,}")
        self.card_generated.set_value(f"{result.nodes_generated:,}")
        self.card_frontier.set_value(f"{result.max_frontier_size:,}")
        self.card_time.set_value(f"{result.elapsed_time_ms:.2f}")

        # Set solution trajectory and render step 0
        self.solution_path = result.solution_path
        self._render_step(0)

        self.status_label.configure(
            text=(
                f"Solved with {algo_name} in {result.elapsed_time_ms:.2f} ms! "
                f"Solution depth d = {result.cost}. Use Playback controls to animate."
            ),
            text_color=COLORS["badge_solvable_fg"],
        )

    def _on_search_error(self, error_msg: str) -> None:
        """Main-thread callback if an unexpected exception occurs during search."""
        self.is_searching = False
        self.solve_btn.configure(state="normal", text="▶  Solve Puzzle")
        self.status_label.configure(
            text=f"Error during search: {error_msg}",
            text_color=COLORS["badge_unsolvable_fg"],
        )

    # -----------------------------------------------------------------
    # Playback & Animation Controls
    # -----------------------------------------------------------------

    def _render_step(self, step_index: int) -> None:
        """Updates the visual 3x3 board and telemetry labels to the specified step index."""
        if not self.solution_path:
            return

        step_index = max(0, min(step_index, len(self.solution_path) - 1))
        self.current_step_index = step_index
        node = self.solution_path[step_index]
        total_moves = len(self.solution_path) - 1

        self.current_state = node.state
        self.board_widget.update_state(node.state)

        self.step_counter_label.configure(text=f"Step {step_index} / {total_moves}")
        action_str = node.action if node.action is not None else "Initial"
        self.node_info_label.configure(
            text=(
                f"Action: {action_str}  |  "
                f"g(n) = {node.g}  |  "
                f"h(n) = {node.h:.0f}  |  "
                f"f(n) = {node.f:.0f}"
            )
        )
        self._update_playback_buttons()

    def _on_play_clicked(self) -> None:
        """Starts automatic step-by-step animation along the solution path."""
        if not self.solution_path or self.is_playing:
            return

        # If already at the goal state, restart animation from step 0
        if self.current_step_index >= len(self.solution_path) - 1:
            self._render_step(0)

        self.is_playing = True
        self._update_playback_buttons()
        self._schedule_next_frame()

    def _schedule_next_frame(self) -> None:
        """Schedules the next animation step using Tkinter's non-blocking after() timer."""
        if not self.is_playing:
            return

        delay_ms = int(self.step_delay_seconds * 1000)
        self.animation_after_id = self.after(delay_ms, self._advance_animation_frame)

    def _advance_animation_frame(self) -> None:
        """Callback that advances one step forward during animation playback."""
        if not self.is_playing or not self.solution_path:
            return

        if self.current_step_index < len(self.solution_path) - 1:
            self._render_step(self.current_step_index + 1)
            if self.current_step_index < len(self.solution_path) - 1:
                self._schedule_next_frame()
            else:
                self._stop_animation()
                self.status_label.configure(
                    text="✓ Goal state reached!",
                    text_color=COLORS["badge_solvable_fg"],
                )
        else:
            self._stop_animation()

    def _on_pause_clicked(self) -> None:
        """Pauses an active solution animation."""
        self._stop_animation()

    def _stop_animation(self) -> None:
        """Cancels any scheduled animation timer and resets playing state."""
        self.is_playing = False
        if self.animation_after_id is not None:
            try:
                self.after_cancel(self.animation_after_id)
            except ValueError:
                pass
            self.animation_after_id = None
        self._update_playback_buttons()

    def _on_step_forward(self) -> None:
        """Steps forward by 1 move along the solution path."""
        self._stop_animation()
        if self.solution_path and self.current_step_index < len(self.solution_path) - 1:
            self._render_step(self.current_step_index + 1)

    def _on_step_backward(self) -> None:
        """Steps backward by 1 move along the solution path."""
        self._stop_animation()
        if self.solution_path and self.current_step_index > 0:
            self._render_step(self.current_step_index - 1)

    def _on_reset_to_initial(self) -> None:
        """Resets the board to Step 0 (the initial state of the solved path)."""
        self._stop_animation()
        if self.solution_path:
            self._render_step(0)

    def _on_speed_changed(self, value: float) -> None:
        """Updates the step delay (0.05s to 1.0s) when the slider is moved."""
        self.step_delay_seconds = float(value)
        self.speed_value_label.configure(text=f"{self.step_delay_seconds:.2f} s / step")

    def _disable_all_playback_buttons(self) -> None:
        """Disables all playback controls while a search is running."""
        for btn in (
            self.btn_reset,
            self.btn_step_back,
            self.btn_play,
            self.btn_pause,
            self.btn_step_fwd,
        ):
            btn.configure(state="disabled")

    def _update_playback_buttons(self) -> None:
        """Enables/disables playback buttons based on current step and playing status."""
        has_solution = len(self.solution_path) > 1
        if not has_solution:
            self._disable_all_playback_buttons()
            return

        at_start = self.current_step_index == 0
        at_end = self.current_step_index >= len(self.solution_path) - 1

        if self.is_playing:
            self.btn_play.configure(state="disabled")
            self.btn_pause.configure(state="normal")
            self.btn_step_back.configure(state="disabled")
            self.btn_step_fwd.configure(state="disabled")
            self.btn_reset.configure(state="normal")
        else:
            self.btn_play.configure(state="normal")
            self.btn_pause.configure(state="disabled")
            self.btn_reset.configure(state="disabled" if at_start else "normal")
            self.btn_step_back.configure(state="disabled" if at_start else "normal")
            self.btn_step_fwd.configure(state="disabled" if at_end else "normal")


def main() -> None:
    """Entry point to launch the 8-Puzzle CustomTkinter desktop application."""
    app = EightPuzzleGUI()
    app.mainloop()


if __name__ == "__main__":
    main()

