"""
test_puzzle.py: Unit tests for core puzzle representation, state transitions,
inversion counts, solvability checks, and node mechanics.
"""

import pytest
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


def test_goal_state_properties():
    assert GOAL_STATE == (1, 2, 3, 4, 5, 6, 7, 8, 0)
    assert len(GOAL_STATE) == 9
    assert count_inversions(GOAL_STATE) == 0
    assert is_solvable(GOAL_STATE) is True


def test_node_ordering_and_tie_breaking():
    # Primary tie-breaking: lower f
    n1 = Node(state=GOAL_STATE, g=1, h=2, f=3)
    n2 = Node(state=GOAL_STATE, g=2, h=3, f=5)
    assert n1 < n2

    # Secondary tie-breaking: same f, lower h preferred (closer to goal)
    n3 = Node(state=GOAL_STATE, g=3, h=1, f=4)
    n4 = Node(state=GOAL_STATE, g=2, h=2, f=4)
    assert n3 < n4

    # Tertiary tie-breaking: insertion order (creation_id)
    n5 = Node(state=GOAL_STATE, g=2, h=2, f=4)
    assert n4 < n5  # n4 was created before n5


def test_node_extract_path():
    root = Node(state=(1, 2, 3, 4, 5, 6, 7, 0, 8), g=0)
    child = Node(state=GOAL_STATE, parent=root, action="RIGHT", g=1)

    path = child.extract_path()
    assert len(path) == 2
    assert path[0] == root
    assert path[1] == child
    assert path[1].action == "RIGHT"


def test_apply_action_valid():
    # Blank in center: (row 1, col 1)
    state = (1, 2, 3, 4, 0, 5, 6, 7, 8)

    up_state = apply_action(state, "UP")
    assert up_state == (1, 0, 3, 4, 2, 5, 6, 7, 8)

    down_state = apply_action(state, "DOWN")
    assert down_state == (1, 2, 3, 4, 7, 5, 6, 0, 8)

    left_state = apply_action(state, "LEFT")
    assert left_state == (1, 2, 3, 0, 4, 5, 6, 7, 8)

    right_state = apply_action(state, "RIGHT")
    assert right_state == (1, 2, 3, 4, 5, 0, 6, 7, 8)


def test_apply_action_invalid_boundary():
    # Blank in top-left corner (0, 0): cannot go UP or LEFT
    corner_state = (0, 1, 2, 3, 4, 5, 6, 7, 8)

    with pytest.raises(ValueError):
        apply_action(corner_state, "UP")

    with pytest.raises(ValueError):
        apply_action(corner_state, "LEFT")

    # DOWN and RIGHT should succeed
    assert apply_action(corner_state, "DOWN") == (3, 1, 2, 0, 4, 5, 6, 7, 8)
    assert apply_action(corner_state, "RIGHT") == (1, 0, 2, 3, 4, 5, 6, 7, 8)


def test_get_successors():
    # Corner has 2 legal successors
    corner = (0, 1, 2, 3, 4, 5, 6, 7, 8)
    succ_corner = get_successors(corner)
    assert len(succ_corner) == 2
    actions = [act for act, _ in succ_corner]
    assert "DOWN" in actions and "RIGHT" in actions

    # Edge has 3 legal successors
    edge = (1, 0, 2, 3, 4, 5, 6, 7, 8)
    succ_edge = get_successors(edge)
    assert len(succ_edge) == 3

    # Center has 4 legal successors
    center = (1, 2, 3, 4, 0, 5, 6, 7, 8)
    succ_center = get_successors(center)
    assert len(succ_center) == 4


def test_inversions_and_solvability():
    # Goal state has 0 inversions -> solvable
    assert count_inversions(GOAL_STATE) == 0
    assert is_solvable(GOAL_STATE) is True

    # State with 1 inversion (swap 7 and 8): odd parity -> unsolvable
    unsolvable = (1, 2, 3, 4, 5, 6, 8, 7, 0)
    assert count_inversions(unsolvable) == 1
    assert is_solvable(unsolvable) is False

    # State with 2 inversions (swap 5 and 6, swap 7 and 8): even parity -> solvable
    solvable_2 = (1, 2, 3, 4, 6, 5, 8, 7, 0)
    assert count_inversions(solvable_2) == 2
    assert is_solvable(solvable_2) is True


def test_parse_board():
    # Valid string formats
    s1 = parse_board("1 2 3 4 5 6 7 8 0")
    assert s1 == GOAL_STATE

    s2 = parse_board("[1, 2, 3, 4, 5, 6, 7, 8, 0]")
    assert s2 == GOAL_STATE

    s3 = parse_board([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
    assert s3 == GOAL_STATE

    # Invalid lengths or values
    with pytest.raises(ValueError):
        parse_board("1 2 3 4 5")

    with pytest.raises(ValueError):
        parse_board("1 2 3 4 5 6 7 8 9")  # 9 instead of 0

    with pytest.raises(ValueError):
        parse_board("1 1 2 3 4 5 6 7 8")  # duplicate 1


def test_format_board():
    formatted = format_board(GOAL_STATE)
    assert "1 │ 2 │ 3" in formatted
    assert "4 │ 5 │ 6" in formatted
    assert "7 │ 8 │  " in formatted

