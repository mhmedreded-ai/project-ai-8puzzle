"""
main.py: 8-Puzzle Solver & Search Benchmark CLI Application.

Provides an interactive and command-line benchmark tool comparing:
  - Breadth-First Search (BFS) [Uninformed baseline]
  - Greedy Best-First Search [h1: Misplaced Tiles]
  - Greedy Best-First Search [h2: Manhattan Distance]
  - A* Search [h1: Misplaced Tiles]
  - A* Search [h2: Manhattan Distance]

Academic Focus:
  - Inversion parity solvability verification
  - Optimality vs Suboptimality
  - Heuristic admissibility and dominance comparison (h2 >= h1)
  - Search space explosion metrics (Nodes Expanded, Time, Max Frontier)
"""

from __future__ import annotations
import argparse
import sys
import time
from typing import Sequence, Optional

from src.puzzle import (
    GOAL_STATE,
    State,
    Node,
    is_solvable,
    count_inversions,
    format_board,
    parse_board,
)
from src.heuristics import misplaced_tiles, manhattan_distance
from src.search import a_star_search, greedy_search, bfs_search, SearchResult

# Optional tabulate support with standard library fallback
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


# Curated presets with verified optimal depths
PRESETS: dict[str, dict[str, object]] = {
    "easy": {
        "name": "Easy (Optimal Depth 4)",
        "state": (1, 5, 2, 4, 0, 3, 7, 8, 6),
        "desc": "Shallow depth (d=4). Ideal for quick conceptual verification.",
    },
    "medium": {
        "name": "Medium (Optimal Depth 12)",
        "state": (5, 8, 2, 1, 7, 3, 0, 4, 6),
        "desc": "Moderate depth (d=12). Clearly exposes Greedy suboptimality.",
    },
    "hard": {
        "name": "Hard (Optimal Depth 20)",
        "state": (8, 7, 0, 5, 4, 2, 1, 6, 3),
        "desc": "Deep branch (d=20). Demonstrates heuristic dominance and BFS explosion.",
    },
    "unsolvable": {
        "name": "Unsolvable Board (Inversion Parity Test)",
        "state": (1, 2, 3, 4, 5, 6, 8, 7, 0),
        "desc": "Odd parity (1 inversion). Rejection without search expansion.",
    },
}


def render_table(headers: list[str], rows: list[list[object]]) -> str:
    """
    Renders a formatted table using tabulate if installed, or built-in ASCII formatter.
    """
    if HAS_TABULATE:
        return tabulate(rows, headers=headers, tablefmt="fancy_grid")

    # Pure standard library ASCII table fallback
    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(str(cell)))

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    header_str = (
        "| "
        + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        + " |"
    )

    row_strs = []
    for row in rows:
        row_str = (
            "| "
            + " | ".join(
                str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)
            )
            + " |"
        )
        row_strs.append(row_str)

    return "\n".join([sep, header_str, sep] + row_strs + [sep])


def print_banner() -> None:
    """Prints the project banner and academic information."""
    banner = """
========================================================================
       8-PUZZLE SOLVER & ACADEMIC SEARCH BENCHMARK SUITE
       Artificial Intelligence: State Space & Informed Search
========================================================================
Problem Formulation (5-Tuple):
  * State Space: Permutations of {0..8} on a 3x3 grid (0 = blank)
  * Actions: {'UP', 'DOWN', 'LEFT', 'RIGHT'} (blank tile transitions)
  * Transition Model: Result(s, a) -> Valid grid swap
  * Goal Test: s == (1, 2, 3, 4, 5, 6, 7, 8, 0)
  * Step Cost: c(s, a, s') = 1 -> Path cost g(n) = tree depth
========================================================================
"""
    print(banner)


def check_and_display_solvability(state: State) -> bool:
    """
    Evaluates and displays the mathematical solvability of the state.

    Returns:
        bool: True if solvable, False otherwise.
    """
    inversions = count_inversions(state)
    solvable = is_solvable(state)
    blank_idx = state.index(0)
    blank_row, blank_col = divmod(blank_idx, 3)

    print("\n--- [Solvability Verification] ---")
    print(f"Board State: {state}")
    print(f"Blank Tile (0) at Position: row={blank_row}, col={blank_col} (index {blank_idx})")
    print(f"Inversion Count: {inversions} ({'Even' if inversions % 2 == 0 else 'Odd'})")
    print(
        f"Goal Inversion Count: {count_inversions(GOAL_STATE)} (Canonical Goal Parity: Even)"
    )

    if solvable:
        print("[✓] SOLVABILITY STATUS: REACHABLE")
        print(
            "    Mathematical Invariant: Legal grid shifts preserve inversion parity mod 2.\n"
            "    Since inversion count parity matches the goal (0 mod 2), this configuration\n"
            "    is proven to have at least one valid path to the goal state."
        )
    else:
        print("[✗] SOLVABILITY STATUS: UNSOLVABLE (PRUNED)")
        print(
            "    Mathematical Proof: Every legal move modifies the inversion count by\n"
            "    0, +2, or -2. Thus, the parity of inversions (inversions % 2) is strictly\n"
            "    invariant. Because the goal parity is 0 and this state has parity 1 (odd),\n"
            "    reaching the goal is topologically impossible. Search aborted."
        )

    return solvable


def run_benchmark(state: State) -> list[SearchResult]:
    """
    Runs the five core search algorithms on the specified initial state.
    """
    algorithms = [
        ("BFS (Uninformed)", None, lambda: bfs_search(state)),
        (
            "Greedy Best-First",
            "Misplaced (h1)",
            lambda: greedy_search(state, misplaced_tiles),
        ),
        (
            "Greedy Best-First",
            "Manhattan (h2)",
            lambda: greedy_search(state, manhattan_distance),
        ),
        ("A* Search", "Misplaced (h1)", lambda: a_star_search(state, misplaced_tiles)),
        ("A* Search", "Manhattan (h2)", lambda: a_star_search(state, manhattan_distance)),
    ]

    results: list[SearchResult] = []
    print("\nExecuting benchmark across all search strategies...")

    for alg_name, heur_name, search_fn in algorithms:
        h_str = f" [{heur_name}]" if heur_name else ""
        print(f"  -> Running {alg_name}{h_str}...", end="", flush=True)
        res = search_fn()
        results.append(res)
        status_symbol = "✓" if res.success else "✗"
        print(
            f" Done ({status_symbol}) in {res.elapsed_time_ms:.2f} ms | Cost: {res.cost} | Expanded: {res.nodes_expanded}"
        )

    return results


def display_benchmark_table(results: list[SearchResult]) -> None:
    """
    Formats and prints the comparative benchmark table.
    """
    # Determine the true optimal cost from A* (since A* with consistent heuristic is optimal)
    optimal_costs = [
        res.cost for res in results if res.success and "A*" in res.algorithm
    ]
    optimal_cost = optimal_costs[0] if optimal_costs else None

    headers = [
        "Algorithm",
        "Heuristic",
        "Status",
        "Path Cost (d)",
        "Nodes Expanded",
        "Nodes Generated",
        "Max Frontier",
        "Time (ms)",
        "Optimal?",
    ]

    rows = []
    for res in results:
        h_display = res.heuristic_name if res.heuristic_name else "None (Uninformed)"
        status_str = "Solved" if res.success else "Failed"
        cost_str = str(res.cost) if res.success else "N/A"
        is_opt = (
            "YES"
            if (optimal_cost is not None and res.cost == optimal_cost and res.success)
            else ("NO" if res.success else "N/A")
        )

        rows.append(
            [
                res.algorithm,
                h_display,
                status_str,
                cost_str,
                f"{res.nodes_expanded:,}",
                f"{res.nodes_generated:,}",
                f"{res.max_frontier_size:,}",
                f"{res.elapsed_time_ms:.2f}",
                is_opt,
            ]
        )

    print("\n=== SEARCH STRATEGY BENCHMARK RESULTS ===")
    print(render_table(headers, rows))


def display_academic_insights(results: list[SearchResult]) -> None:
    """
    Generates comparative academic analysis based on empirical results.
    """
    by_key: dict[str, SearchResult] = {}
    for r in results:
        key = f"{r.algorithm}_{r.heuristic_name}"
        by_key[key] = r

    a_star_h1 = by_key.get("A* Search_misplaced_tiles")
    a_star_h2 = by_key.get("A* Search_manhattan_distance")
    greedy_h2 = by_key.get("Greedy Best-First_manhattan_distance")
    bfs_res = by_key.get("BFS_None")

    print("\n=== ACADEMIC ANALYSIS & THEORETICAL VERIFICATION ===")

    # 1. Heuristic Dominance (h2 >= h1)
    if a_star_h1 and a_star_h2 and a_star_h1.success and a_star_h2.success:
        print("\n1. Heuristic Dominance Proof (h2 dominates h1):")
        print("   - Mathematical condition: For all states n, h2(n) >= h1(n).")
        print(
            f"   - Nodes Expanded with h1 (Misplaced Tiles):     {a_star_h1.nodes_expanded:,}"
        )
        print(
            f"   - Nodes Expanded with h2 (Manhattan Distance):  {a_star_h2.nodes_expanded:,}"
        )
        if a_star_h1.nodes_expanded > 0:
            reduction = (
                (a_star_h1.nodes_expanded - a_star_h2.nodes_expanded)
                / a_star_h1.nodes_expanded
                * 100
            )
            ratio = (
                a_star_h1.nodes_expanded / max(a_star_h2.nodes_expanded, 1)
            )
            print(
                f"   [Observation]: A*(h2) expanded {ratio:.2f}x fewer nodes ({reduction:.1f}% search space reduction)!"
            )
            print(
                "   [Theorem]: Since h2 dominates h1 and both are consistent, A*(h2) is guaranteed\n"
                "   to expand a subset of nodes expanded by A*(h1) up to frontier tie-breaking."
            )

    # 2. Optimality Comparison: A* vs Greedy
    if a_star_h2 and greedy_h2 and a_star_h2.success and greedy_h2.success:
        print("\n2. Optimality vs Greediness Trade-off:")
        print(f"   - A*(h2) Path Cost:      {a_star_h2.cost} moves (Guaranteed Optimal)")
        print(f"   - Greedy(h2) Path Cost:  {greedy_h2.cost} moves")
        if greedy_h2.cost > a_star_h2.cost:
            diff = greedy_h2.cost - a_star_h2.cost
            print(
                f"   [Observation]: Greedy was suboptimal by {diff} additional move(s) ({greedy_h2.cost} vs {a_star_h2.cost}).\n"
                "   [Explanation]: Greedy minimizes h(n) without considering path cost g(n),\n"
                "   making it vulnerable to deceptive local optima and path detours."
            )
        else:
            print(
                "   [Observation]: Greedy matched the optimal cost on this particular instance,\n"
                "   though optimality is NOT theoretically guaranteed in general."
            )

    # 3. Uninformed vs Informed Search (BFS vs A*)
    if bfs_res and a_star_h2 and bfs_res.success and a_star_h2.success:
        print("\n3. Informed Search vs Uninformed Baseline (BFS vs A*):")
        print(f"   - BFS Nodes Expanded:           {bfs_res.nodes_expanded:,}")
        print(f"   - A*(h2) Nodes Expanded:        {a_star_h2.nodes_expanded:,}")
        if a_star_h2.nodes_expanded > 0:
            factor = bfs_res.nodes_expanded / max(a_star_h2.nodes_expanded, 1)
            print(
                f"   [Observation]: A*(Manhattan) pruned the search space by a factor of {factor:.1f}x over BFS!"
            )
            print(
                "   [Complexity]: BFS exhibits O(b^d) space and time complexity, whereas A* guided by\n"
                "   an admissible heuristic focuses exploration along the most promising contours."
            )


def animate_solution(
    solution_path: list[Node],
    delay: float = 0.35,
    interactive_step: bool = False,
) -> None:
    """
    Animates or steps through the board states along the solution trajectory.
    """
    if not solution_path:
        print("No solution path to display.")
        return

    total_steps = len(solution_path) - 1
    print(f"\n=== SOLUTION PATH VISUALIZATION (Total Moves: {total_steps}) ===")
    print(f"Initial State:\n{format_board(solution_path[0].state)}")

    for step_num, node in enumerate(solution_path[1:], start=1):
        if interactive_step:
            try:
                input(f"\n[Press Enter for Step {step_num}/{total_steps} (Action: {node.action})...]")
            except (KeyboardInterrupt, EOFError):
                print("\nAnimation interrupted by user.")
                break
        else:
            time.sleep(delay)

        print(
            f"\nStep {step_num}/{total_steps} | Action: {node.action} | "
            f"g={node.g}, h={node.h:.1f}, f={node.f:.1f}"
        )
        print(format_board(node.state))

    print("\n[✓] Goal State Successfully Reached!")


def interactive_menu() -> None:
    """Provides an interactive CLI menu when no command line arguments are given."""
    print_banner()

    while True:
        print("\nAvailable Options:")
        print("  1. Run Preset: Easy (Depth 4)")
        print("  2. Run Preset: Medium (Depth 12)")
        print("  3. Run Preset: Hard (Depth 20)")
        print("  4. Run Preset: Unsolvable (Inversion Test)")
        print("  5. Enter Custom 3x3 Board")
        print("  6. Benchmark All Presets Sequentially")
        print("  7. Exit")

        choice = input("\nEnter your choice (1-7): ").strip()

        if choice == "1":
            target_state = PRESETS["easy"]["state"]
            execute_workflow(target_state, title="Preset: Easy (Depth 4)")
        elif choice == "2":
            target_state = PRESETS["medium"]["state"]
            execute_workflow(target_state, title="Preset: Medium (Depth 12)")
        elif choice == "3":
            target_state = PRESETS["hard"]["state"]
            execute_workflow(target_state, title="Preset: Hard (Depth 20)")
        elif choice == "4":
            target_state = PRESETS["unsolvable"]["state"]
            execute_workflow(target_state, title="Preset: Unsolvable")
        elif choice == "5":
            print("\nEnter 9 integers (0 through 8) separated by spaces.")
            print("Example: 1 5 2 4 0 3 7 8 6")
            user_in = input("Board input: ").strip()
            try:
                parsed = parse_board(user_in)
                execute_workflow(parsed, title="Custom Board")
            except Exception as e:
                print(f"[!] Error parsing board: {e}")
        elif choice == "6":
            for p_key, p_val in PRESETS.items():
                print(f"\n=======================================================")
                print(f"BENCHMARKING PRESET: {p_val['name']}")
                print(f"Description: {p_val['desc']}")
                print(f"=======================================================")
                execute_workflow(
                    p_val["state"],  # type: ignore
                    title=str(p_val["name"]),
                    prompt_animation=False,
                )
        elif choice == "7":
            print("Exiting. Thank you!")
            break
        else:
            print("Invalid choice, please select between 1 and 7.")


def execute_workflow(
    state: State,
    title: str = "Board Evaluation",
    prompt_animation: bool = True,
    auto_animate: bool = False,
    show_solution: bool = False,
    delay: float = 0.3,
) -> None:
    """Executes the complete evaluation workflow for a given board state."""
    print(f"\n-------------------------------------------------------------")
    print(f"--- {title} ---")
    print(f"-------------------------------------------------------------")
    print("Initial Board Configuration:")
    print(format_board(state))

    solvable = check_and_display_solvability(state)
    if not solvable:
        return

    results = run_benchmark(state)
    display_benchmark_table(results)
    display_academic_insights(results)

    # Find the optimal solution path (A* with Manhattan)
    best_result = next(
        (r for r in results if r.algorithm == "A* Search" and r.heuristic_name == "manhattan_distance"),
        results[-1],
    )

    if auto_animate:
        animate_solution(best_result.solution_path, delay=delay)
    elif show_solution:
        animate_solution(best_result.solution_path, delay=0.0)
    elif prompt_animation and best_result.solution_path and sys.stdin.isatty():
        try:
            anim_choice = input("\nWould you like to animate the optimal solution path? [y/N]: ").strip().lower()
            if anim_choice in ("y", "yes"):
                step_mode = input("Step-by-step interactive mode? [y/N]: ").strip().lower() in ("y", "yes")
                animate_solution(best_result.solution_path, delay=delay, interactive_step=step_mode)
        except (KeyboardInterrupt, EOFError):
            pass


def parse_cli_args() -> argparse.Namespace:
    """Configures and parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="8-Puzzle Solver & Academic Search Benchmark Suite (Russell & Norvig Aligned)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --preset easy
  python main.py --preset hard --animate
  python main.py --custom "1 5 2 4 0 3 7 8 6"
  python main.py --all-presets
  python main.py --interactive
        """,
    )
    parser.add_argument(
        "-p",
        "--preset",
        choices=["easy", "medium", "hard", "unsolvable"],
        help="Run benchmark on a standard curated test board.",
    )
    parser.add_argument(
        "-c",
        "--custom",
        type=str,
        help="Custom board state as 9 numbers (e.g. '1 5 2 4 0 3 7 8 6').",
    )
    parser.add_argument(
        "-a",
        "--animate",
        action="store_true",
        help="Automatically animate the optimal solution path in the terminal.",
    )
    parser.add_argument(
        "-s",
        "--solution",
        action="store_true",
        help="Print the full step-by-step solution path without delay.",
    )
    parser.add_argument(
        "-d",
        "--delay",
        type=float,
        default=0.3,
        help="Animation delay between steps in seconds (default: 0.3s).",
    )
    parser.add_argument(
        "--all-presets",
        action="store_true",
        help="Execute the benchmark across all presets sequentially.",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Launch the interactive menu.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_cli_args()

    # If no flags provided or --interactive specified, launch interactive menu
    if len(sys.argv) == 1 or args.interactive:
        interactive_menu()
        return

    print_banner()

    if args.all_presets:
        for p_key, p_val in PRESETS.items():
            print(f"\n=======================================================")
            print(f"BENCHMARKING PRESET: {p_val['name']}")
            print(f"Description: {p_val['desc']}")
            print(f"=======================================================")
            execute_workflow(
                p_val["state"],  # type: ignore
                title=str(p_val["name"]),
                prompt_animation=False,
                auto_animate=args.animate,
                show_solution=args.solution,
                delay=args.delay,
            )
        return

    if args.preset:
        preset_info = PRESETS[args.preset]
        execute_workflow(
            preset_info["state"],  # type: ignore
            title=str(preset_info["name"]),
            prompt_animation=not (args.animate or args.solution),
            auto_animate=args.animate,
            show_solution=args.solution,
            delay=args.delay,
        )
        return

    if args.custom:
        try:
            custom_state = parse_board(args.custom)
            execute_workflow(
                custom_state,
                title="Custom Board",
                prompt_animation=not (args.animate or args.solution),
                auto_animate=args.animate,
                show_solution=args.solution,
                delay=args.delay,
            )
        except Exception as e:
            print(f"[!] Error with custom board: {e}")
            sys.exit(1)
        return


if __name__ == "__main__":
    main()
