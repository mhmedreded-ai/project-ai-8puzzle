"""
test_search.py: Unit tests for BFS, Greedy Best-First, and A* Search algorithms.
Verifies optimality, path validity, metrics tracking, and unsolvable state pruning.
"""

from src.puzzle import GOAL_STATE, apply_action
from src.heuristics import misplaced_tiles, manhattan_distance
from src.search import a_star_search, greedy_search, bfs_search, SearchResult


def is_path_valid(initial_state, solution_path):
    """Verifies that each step in solution_path executes a valid legal transition."""
    if not solution_path:
        return False
    if solution_path[0].state != initial_state:
        return False
    if solution_path[-1].state != GOAL_STATE:
        return False

    current = initial_state
    for node in solution_path[1:]:
        assert node.action is not None
        current = apply_action(current, node.action)
        assert current == node.state

    return current == GOAL_STATE


def test_search_already_at_goal():
    res_bfs = bfs_search(GOAL_STATE)
    assert res_bfs.success is True
    assert res_bfs.cost == 0
    assert len(res_bfs.solution_path) == 1

    res_astar = a_star_search(GOAL_STATE, manhattan_distance)
    assert res_astar.success is True
    assert res_astar.cost == 0

    res_greedy = greedy_search(GOAL_STATE, manhattan_distance)
    assert res_greedy.success is True
    assert res_greedy.cost == 0


def test_tuple_unpacking_interface():
    """Verify requirement that search functions support (path, nodes_expanded, elapsed_time) unpacking."""
    easy_state = (1, 5, 2, 4, 0, 3, 7, 8, 6)
    path, nodes_expanded, elapsed_time = a_star_search(easy_state, manhattan_distance)

    assert isinstance(path, list)
    assert isinstance(nodes_expanded, int)
    assert isinstance(elapsed_time, float)
    assert nodes_expanded > 0
    assert elapsed_time >= 0.0


def test_unsolvable_rejection():
    unsolvable = (1, 2, 3, 4, 5, 6, 8, 7, 0)

    res_bfs = bfs_search(unsolvable)
    assert res_bfs.success is False
    assert res_bfs.nodes_expanded == 0

    res_astar = a_star_search(unsolvable, manhattan_distance)
    assert res_astar.success is False
    assert res_astar.nodes_expanded == 0

    res_greedy = greedy_search(unsolvable, manhattan_distance)
    assert res_greedy.success is False
    assert res_greedy.nodes_expanded == 0


def test_search_optimality_and_path_validity():
    # Easy preset: depth 4
    state = (1, 5, 2, 4, 0, 3, 7, 8, 6)

    # 1. BFS (Guaranteed Optimal for unit step cost)
    bfs_res = bfs_search(state)
    assert bfs_res.success is True
    assert bfs_res.cost == 4
    assert is_path_valid(state, bfs_res.solution_path)

    # 2. A* with Misplaced Tiles (Guaranteed Optimal)
    astar_h1 = a_star_search(state, misplaced_tiles)
    assert astar_h1.success is True
    assert astar_h1.cost == 4
    assert is_path_valid(state, astar_h1.solution_path)

    # 3. A* with Manhattan Distance (Guaranteed Optimal)
    astar_h2 = a_star_search(state, manhattan_distance)
    assert astar_h2.success is True
    assert astar_h2.cost == 4
    assert is_path_valid(state, astar_h2.solution_path)

    # 4. Greedy Search
    greedy_res = greedy_search(state, manhattan_distance)
    assert greedy_res.success is True
    assert is_path_valid(state, greedy_res.solution_path)


def test_dominance_in_search():
    """
    On a non-trivial depth-12 board, A*(h2) must expand fewer or equal nodes than A*(h1).
    """
    state_d12 = (5, 8, 2, 1, 7, 3, 0, 4, 6)

    astar_h1 = a_star_search(state_d12, misplaced_tiles)
    astar_h2 = a_star_search(state_d12, manhattan_distance)

    assert astar_h1.success is True
    assert astar_h2.success is True
    assert astar_h1.cost == astar_h2.cost == 12

    # Manhattan dominance ensures search space pruning
    assert astar_h2.nodes_expanded <= astar_h1.nodes_expanded

