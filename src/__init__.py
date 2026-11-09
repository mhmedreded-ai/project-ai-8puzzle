"""
8-Puzzle Solver & Search Benchmark Package.

This package provides an academically aligned framework for formulating,
solving, and benchmarking the 8-puzzle problem using uninformed and informed
heuristic search strategies.
"""

from src.puzzle import (
    GOAL_STATE,
    State,
    Node,
    is_solvable,
    count_inversions,
    get_successors,
    apply_action,
    format_board,
    parse_board,
)
from src.heuristics import misplaced_tiles, manhattan_distance
from src.search import a_star_search, greedy_search, bfs_search, SearchResult

__all__ = [
    "GOAL_STATE",
    "State",
    "Node",
    "is_solvable",
    "count_inversions",
    "get_successors",
    "apply_action",
    "format_board",
    "parse_board",
    "misplaced_tiles",
    "manhattan_distance",
    "a_star_search",
    "greedy_search",
    "bfs_search",
    "SearchResult",
]

