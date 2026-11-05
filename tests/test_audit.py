import pandas as pd

from receval.audit import duplicate_detail, long_view_rule, valid_play


def test_valid_play_follows_the_readme_boundaries():
    frame = pd.DataFrame({"duration_ms": [5_000, 5_000, 20_000, 20_000],
                          "play_time_ms": [5_000, 4_999, 7_001, 7_000]})
    assert valid_play(frame).tolist() == [1, 0, 1, 0]


def test_long_view_follows_the_readme_boundaries():
    frame = pd.DataFrame({"duration_ms": [10_000, 10_000, 30_000, 30_000],
                          "play_time_ms": [10_000, 9_999, 18_000, 17_999]})
    assert long_view_rule(frame).tolist() == [1, 0, 1, 0]


def test_duplicate_detail_separates_identical_and_conflicting_rows():
    frame = pd.DataFrame({"user_id": [1, 1, 2, 2], "video_id": [9, 9, 8, 8], "time_ms": [5, 5, 6, 6],
                          "is_click": [1, 1, 0, 1], "long_view": [0, 0, 0, 0], "play_time_ms": [3, 3, 4, 4]})
    assert duplicate_detail(frame) == {"groups": 2, "groups_with_differing_feedback": 1}
