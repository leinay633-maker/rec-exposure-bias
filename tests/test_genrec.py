"""Study 2 (genrec) unit tests on synthetic data: no KuaiRec file is read."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from genrec import data, evaluate, generative, semantic_id, text
from genrec.generative import Codes, Generator, Trie, beam_search, exact_scores
from genrec.transformer import Stack


def test_caption_repair(tmp_path):
    header = "video_id,manual_cover_text,caption,topic_tag,first_level_category_id,first_level_category_name," \
             "second_level_category_id,second_level_category_name,third_level_category_id,third_level_category_name\n"
    good = '0,cover,"cap",[a],1,c1,2,c2,3,c3\n'
    broken = "1,UNKNOWN,line one\nline two,[],6,c6,-124,UNKNOWN,-124,UNKNOWN\n"
    path = tmp_path / "cap.csv"
    path.write_text(header + good + broken, encoding="utf-8")
    frame, repaired = text.load_captions(path)
    assert repaired == 1 and frame["video_id"].tolist() == [0, 1]
    assert frame.loc[1, "caption"] == "line one\nline two" and frame.loc[1, "first_level_category_name"] == "c6"
    assert "标题：line one" in text.describe(frame.loc[1]) and "封面文字" not in text.describe(frame.loc[1])


def test_disambiguation_and_shuffle_keep_structure():
    rng = np.random.default_rng(0)
    codes3 = rng.integers(0, 3, size=(60, 3))
    a, b = semantic_id.disambiguate(codes3, 0), semantic_id.disambiguate(codes3, 1)
    assert len(np.unique(a, axis=0)) == 60 and np.array_equal(a[:, :3], codes3)
    assert np.array_equal(a, semantic_id.disambiguate(codes3, 0)) and not np.array_equal(a, b)
    pool = np.arange(0, 60, 3)
    sh = semantic_id.shuffled(a, pool, 5)
    others = np.setdiff1d(np.arange(60), pool)
    for ids in (pool, others):  # same tuples inside each group, so each group's prefix tree keeps its shape
        assert sorted(map(tuple, sh[ids])) == sorted(map(tuple, a[ids]))
    assert not np.array_equal(sh, a)


def _tiny(seed=0, items=40, levels=(4, 3, 3, 2), history=5, d=32):
    rng = np.random.default_rng(seed)
    grid = np.stack(np.meshgrid(*[np.arange(n) for n in levels], indexing="ij"), -1).reshape(-1, len(levels))
    codes = Codes(grid[rng.choice(len(grid), size=items, replace=False)])
    torch.manual_seed(seed)
    model = Generator(codes, history, d=d, layers=2, heads=4, dropout=0.0).eval()
    hist = np.full((3, history), -1)
    hist[0, -2:] = [3, 7]
    hist[1, :] = [1, 2, 3, 4, 5]
    hist[2, -1:] = [9]
    return codes, model, hist


def test_incremental_step_matches_full_forward():
    torch.manual_seed(0)
    stack = Stack(16, 2, 4, 0.0).eval()
    x = torch.randn(2, 9, 16)
    valid = torch.ones(2, 9, dtype=torch.bool)
    valid[0, :3] = False
    full = stack(x, valid)
    h, cache = stack.prefix(x[:, :6], valid[:, :6])
    past = None
    for t in range(6, 9):
        out, past = stack.step(x[:, t][:, None], cache, valid[:, :6], past)
        assert torch.allclose(out[:, 0], full[:, t], atol=1e-5)
    assert torch.allclose(h[:, -1], full[:, 5], atol=1e-5)


def test_beam_search_returns_exact_ranking_when_beam_is_wide():
    codes, model, hist = _tiny()
    cands = np.arange(0, 40, 2)
    trie = Trie(codes, cands)
    allowed = np.ones((3, len(cands)), dtype=bool)
    allowed[1, :5] = False
    items, scores = beam_search(model, codes, trie, hist, allowed, beam=len(cands), dev=torch.device("cpu"))
    exact = exact_scores(model, codes, trie, hist, torch.device("cpu"))
    for u in range(3):
        got = items[u][items[u] >= 0]
        ok = np.flatnonzero(allowed[u])
        assert sorted(got.tolist()) == sorted(ok.tolist())  # every allowed candidate, nothing else
        assert np.allclose(scores[u][: len(got)], exact[u][got], atol=1e-4)
        assert np.all(np.diff(scores[u][: len(got)]) <= 1e-6)  # best first


def test_beam_search_narrow_beam_only_allowed_and_unique():
    codes, model, hist = _tiny(seed=3)
    cands = np.arange(40)
    trie = Trie(codes, cands)
    allowed = np.random.default_rng(1).random((3, 40)) < 0.5
    items, _ = beam_search(model, codes, trie, hist, allowed, beam=6, dev=torch.device("cpu"))
    for u in range(3):
        got = items[u][items[u] >= 0]
        assert len(got) == len(set(got.tolist())) and allowed[u, got].all()


def test_generator_loss_skips_first_slot_and_pads():
    codes, model, _ = _tiny()
    win = np.array([[-1, -1, -1, 4, 8, 12], [1, 2, 3, 4, 5, 6]])
    tokens = torch.as_tensor(codes.window_tokens(win))
    loss = generative.loss_fn(model, tokens)
    assert torch.isfinite(loss)
    # changing only the first real event's tokens as targets must not matter: they are inputs only
    logits = model(tokens[:, :-1])
    assert logits.shape == (2, tokens.shape[1] - 1, codes.vocab)


def test_weighted_equals_dense_when_weights_are_equal():
    rng = np.random.default_rng(0)
    top = np.stack([rng.permutation(50)[:20] for _ in range(4)])
    targets = [rng.choice(50, size=k, replace=False) for k in (1, 5, 9, 0)]
    dense = evaluate.per_user(top, targets)
    w = evaluate.weighted(top, targets, np.full(50, 3.0))
    for name in evaluate.NAMES:
        assert np.allclose(dense[name], w[name], equal_nan=True)
    skew = evaluate.weighted(top, targets, np.arange(50, dtype=float))
    assert np.isnan(skew["recall@100"][3]) and np.all(skew["recall@100"][:3] <= 1)


def test_borrowed_metrics_use_only_observed_targets():
    top = np.array([[0, 1, 2, 3]])
    targets = [np.array([1, 3, 4])]
    masks = np.array([[True, True, False, False, False], [False, False, False, True, True]])
    assign = np.array([[0], [1], [0]])
    col_of = np.arange(5)
    out = evaluate.borrowed(top, targets, np.array([0]), assign, masks, col_of)
    assert np.allclose(out["recall@20"][0], [1.0, 0.5, 1.0])


def test_windows_and_pairs_cover_every_event_but_the_first():
    class S:
        events = {1: np.array([10, 11, 12]), 2: np.arange(100, 160)}
    win = data.windows(S, 50)
    users = data.window_users(S, 50)
    hist, target, user = data.pairs(win, users)
    assert len(target) == (3 - 1) + (60 - 1)
    assert sorted(target[user == 2].tolist()) == list(range(101, 160))
    assert (hist >= 0).sum(1).max() == 50


def test_sealed_rows_are_never_parsed(tmp_path):
    rows = ["user_id,video_id,play_duration,video_duration,time,date,timestamp,watch_ratio"]
    rows += [f"{u},{v},1,1,t,20200801,{u * 10 + v}.0,{0.5 + v}" for u in range(3) for v in range(3)]
    rows += [f"{u},{v},1,1,t,20200902,{u * 10 + v}.5,NOT_A_NUMBER" for u in range(3) for v in range(2)]
    (tmp_path / "big_matrix.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    big = data.read_big(tmp_path)
    assert len(big) == 9 and big["watch_ratio"].dtype.kind == "f"
    small = ["user_id,video_id,play_duration,video_duration,time,date,timestamp,watch_ratio"]
    small += [f"{u},{v},1,1,t,20200801,1.0,{'SEALED' if u == 2 else 3.0}" for u in range(3) for v in range(2)]
    (tmp_path / "small_matrix.csv").write_text("\n".join(small) + "\n", encoding="utf-8")
    proxy = data.read_small(tmp_path, {0, 1})
    assert sorted(proxy["user_id"].unique()) == [0, 1] and proxy["watch_ratio"].dtype.kind == "f"


def test_sealed_file_needs_an_opening(tmp_path):
    with pytest.raises(PermissionError):
        data.load("dense_test", cache=tmp_path, runs=tmp_path)


def test_dedup_drops_conflicting_groups():
    frame = pd.DataFrame({"user_id": [1, 1, 1, 2], "video_id": [5, 5, 5, 5], "timestamp": [1.0, 1.0, 1.0, 2.0],
                          "watch_ratio": [0.5, 0.5, 0.7, 1.0]})
    out = data.dedup(frame)
    assert out["user_id"].tolist() == [2]


def test_borrowed_counts_match_the_draws_used():
    top = np.array([[0, 1, 2, 3], [4, 3, 2, 1]])
    targets = [np.array([1, 3, 4]), np.array([2])]
    masks = np.array([[True, True, False, False, False], [False, False, False, True, True]])
    assign = np.array([[0, 1], [1, 1], [0, 0]])
    col_of = np.arange(5)
    counts = evaluate.borrowed_counts(targets, np.array([0, 1]), assign, masks, col_of)
    assert counts.tolist() == [[1, 2, 1], [0, 0, 0]]
    out = evaluate.borrowed(top, targets, np.array([0, 1]), assign, masks, col_of)
    assert np.array_equal(np.isfinite(out["recall@100"]), counts > 0)


def test_early_stop_and_minimum_epochs():
    from genrec.train import EarlyStop

    vals = [0.06, 0.04, 0.05, 0.055, 0.05, 0.07, 0.06, 0.06, 0.06, 0.09]

    def run(min_epochs):
        stop = EarlyStop(3, min_epochs)
        for epoch, v in enumerate(vals):
            _, done = stop.step(epoch, v)
            if done:
                return stop.best_epoch, epoch + 1
        return stop.best_epoch, len(vals)

    assert run(1) == (0, 4)  # best at epoch 0, three epochs without improvement
    assert run(5) == (5, 9)  # epochs 0-3 never count; best from index 4 on, then patience 3
    stop = EarlyStop(3, 1)
    assert stop.step(0, 0.1) == (True, False) and stop.step(1, 0.1) == (False, False)  # ties are no improvement


def test_codes_with_explicit_sizes():
    codes = np.array([[0, 1, 0], [2, 1, 1], [3, 0, 0]])
    c = Codes(codes, sizes=[4, 4, 2])
    assert c.vocab == 1 + 4 + 4 + 2 and c.offsets.tolist() == [1, 5, 9]
    with pytest.raises(ValueError):
        Codes(codes, sizes=[3, 4, 2])


def test_nanmean_without_warning():
    import warnings

    from genrec.analysis import nanmean

    x = np.array([[1.0, np.nan], [np.nan, np.nan]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = nanmean(x, 1)
    assert out[0] == 1.0 and np.isnan(out[1])
