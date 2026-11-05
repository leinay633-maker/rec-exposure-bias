import numpy as np
import pandas as pd
import pytest

from receval import features as F


def _side(n_users=3, n_videos=4):
    users = pd.DataFrame({"user_id": range(n_users), **{f: "x" for f in F.USER_FIELDS}})
    videos = pd.DataFrame({"video_id": range(n_videos), "author_id": [10 + v % 2 for v in range(n_videos)],
                           "video_type": "NORMAL", "upload_type": "u", "music_type": 1, "tag": "3,4",
                           "video_duration": 20_000, "server_width": 720, "server_height": 1280})
    return users, videos


def _rows(records):
    base = {"hourmin": 1200, "duration_ms": 20_000, "is_rand": 0, "tab": 1, "long_view": 0, "is_like": 0,
            "play_time_ms": 0}
    return pd.DataFrame([{**base, **r} for r in records])


def test_fit_rejects_rows_after_the_training_window():
    users, videos = _side()
    train = _rows([{"user_id": 0, "video_id": 0, "date": 20220422, "time_ms": 1, "is_click": 1}])
    with pytest.raises(ValueError, match="training window"):
        F.fit(train, users, videos)


def test_build_rejects_evaluation_rows_inside_the_training_window():
    users, videos = _side()
    train = _rows([{"user_id": 0, "video_id": 0, "date": 20220410, "time_ms": 1, "is_click": 1}])
    space = F.fit(train, users, videos)
    F.build(train, space, users, videos, is_train=True)
    with pytest.raises(ValueError, match="after the training window"):
        F.build(train, space, users, videos, is_train=False)


def test_training_histories_only_contain_earlier_positives():
    users, videos = _side()
    train = _rows([{"user_id": 0, "video_id": 1, "date": 20220410, "time_ms": 10, "is_click": 1},
                   {"user_id": 0, "video_id": 2, "date": 20220410, "time_ms": 20, "is_click": 1},
                   {"user_id": 0, "video_id": 3, "date": 20220411, "time_ms": 30, "is_click": 0},
                   {"user_id": 0, "video_id": 0, "date": 20220411, "time_ms": 40, "is_click": 1}])
    space = F.fit(train, users, videos)
    built = F.build(train, space, users, videos, is_train=True)
    idx = {v: space.vocab["video_id"][str(v)] for v in range(4)}
    hist = [list(h[h > 0]) for h in built["hist"]]
    assert hist == [[], [idx[1]], [idx[1], idx[2]], [idx[1], idx[2]]]

    later = _rows([{"user_id": 0, "video_id": 3, "date": 20220423, "time_ms": 50, "is_click": 1},
                   {"user_id": 2, "video_id": 3, "date": 20220423, "time_ms": 60, "is_click": 0}])
    evaluated = F.build(later, space, users, videos, is_train=False)
    assert list(evaluated["hist"][0][evaluated["hist"][0] > 0]) == [idx[1], idx[2], idx[0]]
    assert not evaluated["hist"][1].any()


def test_out_of_fold_statistics_ignore_the_rows_own_label():
    users, videos = _side()
    records = [{"user_id": u % 3, "video_id": u % 4, "date": 20220410, "time_ms": u, "is_click": u % 2,
                "long_view": (u // 2) % 2} for u in range(80)]
    train = _rows(records)
    space = F.fit(train, users, videos)
    before = F.build(train, space, users, videos, is_train=True)
    before = before["num"] * space.numeric_std + space.numeric_mean
    flipped = train.copy()
    flipped.loc[17, ["is_click", "long_view"]] = 1 - flipped.loc[17, ["is_click", "long_view"]]
    space2 = F.fit(flipped, users, videos)
    after = F.build(flipped, space2, users, videos, is_train=True)
    after = after["num"] * space2.numeric_std + space2.numeric_mean
    # float32 storage leaves ~1e-8 noise; a leak of the own label would move the rate by ~1/16.
    np.testing.assert_allclose(before[17], after[17], rtol=0, atol=1e-6)


def test_training_rows_must_come_from_the_training_window():
    users, videos = _side()
    train = _rows([{"user_id": 0, "video_id": 0, "date": 20220410, "time_ms": 1, "is_click": 1}])
    space = F.fit(train, users, videos)
    late = _rows([{"user_id": 0, "video_id": 0, "date": 20220422, "time_ms": 1, "is_click": 1}])
    with pytest.raises(ValueError, match="training window"):
        F.build(late, space, users, videos, is_train=True)


def test_unknown_ids_map_to_zero():
    users, videos = _side()
    train = _rows([{"user_id": 0, "video_id": 0, "date": 20220410, "time_ms": 1, "is_click": 1}])
    space = F.fit(train, users, videos)
    later = _rows([{"user_id": 2, "video_id": 3, "date": 20220423, "time_ms": 5, "is_click": 0}])
    with pytest.raises(ValueError, match="training rows first"):
        F.build(later, space, users, videos, is_train=False)
    F.build(train, space, users, videos, is_train=True)
    built = F.build(later, space, users, videos, is_train=False)
    assert built["cat"][0, F.CAT_FIELDS.index("user_id")] == 0
    assert built["cat"][0, F.CAT_FIELDS.index("video_id")] == 0


def test_dedup_keeps_one_identical_row_and_drops_conflicts():
    frame = _rows([{"user_id": 0, "video_id": 0, "date": 20220410, "time_ms": 1, "is_click": 1},
                   {"user_id": 0, "video_id": 0, "date": 20220410, "time_ms": 1, "is_click": 1},
                   {"user_id": 1, "video_id": 0, "date": 20220410, "time_ms": 2, "is_click": 1},
                   {"user_id": 1, "video_id": 0, "date": 20220410, "time_ms": 2, "is_click": 0},
                   {"user_id": 2, "video_id": 1, "date": 20220410, "time_ms": 3, "is_click": 0},
                   {"user_id": 2, "video_id": 2, "date": 20220410, "time_ms": 4, "is_click": 0, "is_like": 1},
                   {"user_id": 2, "video_id": 2, "date": 20220410, "time_ms": 4, "is_click": 0, "is_like": 0},
                   {"user_id": 2, "video_id": 3, "date": 20220410, "time_ms": 5, "is_click": 0, "tab": 1},
                   {"user_id": 2, "video_id": 3, "date": 20220410, "time_ms": 5, "is_click": 0, "tab": 4}])
    out = F.dedup(frame)
    assert sorted(zip(out["user_id"], out["video_id"])) == [(0, 0), (2, 1)]
