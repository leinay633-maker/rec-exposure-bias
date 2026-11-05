"""Study 3 (docs/rlrec/PROTOCOL.md §10): the masked policy, the analytic optimum and RLOO on synthetic data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import chisquare

from genrec.generative import Codes, Generator
from rlrec import analytic, policy, rl
from rlrec.policy import Tree

CPU = torch.device("cpu")


def _tiny(seed=0, items=40, levels=(4, 3, 3, 2), history=5, d=32, users=4):
    rng = np.random.default_rng(seed)
    grid = np.stack(np.meshgrid(*[np.arange(n) for n in levels], indexing="ij"), -1).reshape(-1, len(levels))
    codes = Codes(grid[rng.choice(len(grid), size=items, replace=False)], sizes=list(levels))
    torch.manual_seed(seed)
    model = Generator(codes, history, d=d, layers=2, heads=4, dropout=0.0).eval()
    hist = np.full((users, history), -1)
    for u in range(users):
        k = rng.integers(1, history + 1)
        hist[u, -k:] = rng.choice(items, k, replace=False)
    allowed = rng.random((users, items)) < 0.6
    allowed[0] = True
    allowed[-1] = False
    allowed[-1, rng.choice(items, 3, replace=False)] = True  # a user with very few candidates
    return codes, model, hist, allowed


def _brute(model, codes, hist, allowed):
    """Masked log π of every allowed video by full forward passes, siblings found by scanning the codes."""
    levels = codes.levels
    out = np.full(allowed.shape, -np.inf)
    with torch.no_grad():
        for u in range(len(hist)):
            ok = np.flatnonzero(allowed[u])
            for i in ok:
                seq = torch.as_tensor(np.concatenate([codes.window_tokens(hist[u:u + 1])[0], codes.tokens[i]]))[None]
                logits = model(seq[:, :-1])[0, -levels:].double()
                total = 0.0
                for lv in range(levels):
                    same = ok[(codes.codes[ok, :lv] == codes.codes[i, :lv]).all(1)]
                    kids = np.unique(codes.codes[same, lv]) + codes.offsets[lv]
                    total += float(logits[lv, codes.tokens[i, lv]] - torch.logsumexp(logits[lv, kids], 0))
                out[u, i] = total
    return out


def test_exact_logprobs_match_brute_force_and_normalise():
    codes, model, hist, allowed = _tiny()
    tree = Tree(codes)
    assert len(tree.multi) and len(tree.multi) < tree.size[codes.levels - 1]  # both kinds of last-level prefixes
    exact = policy.exact_logprobs(model, codes, tree, hist, allowed, CPU, chunk=3)
    brute = _brute(model, codes, hist, allowed)
    assert np.array_equal(np.isfinite(exact), allowed)
    assert np.allclose(exact[allowed], brute[allowed], atol=1e-5)
    assert np.allclose(np.log(np.exp(exact.astype(np.float64)).sum(1)), 0.0, atol=1e-5)


def test_sampling_matches_teacher_forcing_and_exact_distribution():
    codes, model, hist, allowed = _tiny(seed=2)
    tree = Tree(codes)
    gen = torch.Generator().manual_seed(0)
    items, logp = policy.sample(model, codes, tree, hist, allowed, 8, CPU, gen)
    assert all(allowed[u, items[u]].all() for u in range(len(items)))  # only candidates
    t = tree.to(CPU)
    al = policy.alive(tree, t, torch.as_tensor(allowed))
    tf = policy.sequence_logprob(model, codes, tree, t, torch.as_tensor(codes.window_tokens(hist)), al,
                                 torch.as_tensor(items)).detach().numpy()
    exact = policy.exact_logprobs(model, codes, tree, hist, allowed, CPU)
    assert np.allclose(logp, tf, atol=1e-5)
    assert np.allclose(tf, np.take_along_axis(exact, items, 1), atol=1e-5)


def test_sampling_frequencies():
    codes, model, hist, allowed = _tiny(seed=4, users=2)
    tree = Tree(codes)
    gen = torch.Generator().manual_seed(1)
    draws = 40_000
    items, _ = policy.sample(model, codes, tree, hist, allowed, draws, CPU, gen)
    exact = np.exp(policy.exact_logprobs(model, codes, tree, hist, allowed, CPU).astype(np.float64))
    for u in range(len(hist)):
        ok = np.flatnonzero(allowed[u])
        seen = np.bincount(items[u], minlength=len(codes.codes))[ok]
        expected = exact[u, ok] / exact[u, ok].sum() * draws
        assert chisquare(seen, expected).pvalue > 1e-3


def test_teacher_forcing_has_gradients():
    codes, model, hist, allowed = _tiny(seed=5)
    tree = Tree(codes)
    t = tree.to(CPU)
    al = policy.alive(tree, t, torch.as_tensor(allowed))
    items, _ = policy.sample(model, codes, tree, hist, allowed, 3, CPU, torch.Generator().manual_seed(0))
    out = policy.sequence_logprob(model, codes, tree, t, torch.as_tensor(codes.window_tokens(hist)), al,
                                  torch.as_tensor(items))
    out.sum().backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad and p.dim() > 1)


def test_tilt_limits_kl_and_bisection():
    rng = np.random.default_rng(0)
    logp0 = np.log(rng.dirichlet(np.ones(12), size=5))
    logp0[:, -2:] = -np.inf  # non-candidates
    logp0 = logp0 - np.log(np.exp(logp0).sum(1, keepdims=True))
    r = rng.random((5, 12))
    assert np.allclose(analytic.tilt(logp0, r, 1e6), logp0, atol=1e-5)  # β -> ∞: back to π0
    sharp = np.exp(analytic.tilt(logp0, r, 1e-4))
    best = np.where(np.isfinite(logp0), r, -1).argmax(1)
    assert np.allclose(sharp[np.arange(5), best], 1.0, atol=1e-6)  # β -> 0: the best candidate
    assert np.isneginf(analytic.tilt(logp0, r, 0.3)[:, -2:]).all()
    beta = 0.2
    lp = analytic.tilt(logp0, r, beta)
    p = np.exp(lp)
    with np.errstate(invalid="ignore"):
        direct = np.where(p > 0, p * (lp - logp0), 0).sum(1)
    assert np.allclose(analytic.kl(lp, logp0), direct)
    z = np.log(np.exp(np.where(np.isfinite(logp0), logp0 + r / beta, -np.inf)).sum(1))
    assert np.allclose(analytic.kl(lp, logp0), (p * np.where(p > 0, r, 0)).sum(1) / beta - z)  # E[r]/β − log Z
    ks = [analytic.mean_kl(logp0, r, b) for b in (0.01, 0.1, 1.0, 10.0)]
    assert all(a > b for a, b in zip(ks, ks[1:]))  # KL falls as β grows
    target = 0.5 * analytic.kl_limit(logp0, r)
    b, k = analytic.bisect_beta(logp0, r, target)
    assert abs(k - target) <= 1e-3 * target and abs(analytic.mean_kl(logp0, r, b) - k) < 1e-12
    with pytest.raises(ValueError):
        analytic.bisect_beta(logp0, r, 2 * analytic.kl_limit(logp0, r))


def test_rloo_gradient_is_unbiased_for_the_kl_regularised_objective():
    torch.manual_seed(0)
    n, lam, s, groups = 6, 0.3, 8, 40_000
    theta = torch.randn(n, dtype=torch.float64, requires_grad=True)
    logp0 = torch.log_softmax(torch.randn(n, dtype=torch.float64), 0)
    r = torch.rand(n, dtype=torch.float64)
    logp = torch.log_softmax(theta, 0)
    objective = (logp.exp() * r).sum() - lam * (logp.exp() * (logp - logp0)).sum()
    (analytic_grad,) = torch.autograd.grad(objective, theta)
    draws = torch.multinomial(logp.detach().exp(), groups * s, replacement=True,
                              generator=torch.Generator().manual_seed(1)).reshape(groups, s)
    lp = torch.log_softmax(theta, 0)[draws]
    loss = rl.rloo_loss(lp, r[draws], logp0[draws], lam)
    (grad,) = torch.autograd.grad(-loss, theta)
    assert torch.allclose(grad, analytic_grad, atol=4e-3)
    adv = rl.advantages(torch.tensor([[1.0, 2.0, 3.0]]))
    assert torch.allclose(adv, torch.tensor([[-1.5, 0.0, 1.5]]))
    assert torch.allclose(rl.advantages(torch.tensor([[1.0, 2.0, 3.0]]), normalise=True), adv / 1.0)


def _feature_world(seed=0, n_users=6, n_videos=9, n_rows=200):
    from receval import features as F
    rng = np.random.default_rng(seed)
    users = pd.DataFrame({"user_id": range(n_users), "is_live_streamer": rng.integers(0, 2, n_users),
                          "is_video_author": rng.integers(0, 2, n_users),
                          "register_days_range": rng.choice(["a", "b", "c"], n_users),
                          **{f"onehot_feat{i}": rng.integers(0, 3, n_users) for i in range(18)}})
    videos = pd.DataFrame({"video_id": range(n_videos), "author_id": rng.integers(10, 14, n_videos),
                           "video_type": rng.choice(["NORMAL", "AD"], n_videos), "upload_type": rng.choice(["u", "v"], n_videos),
                           "music_type": rng.integers(0, 3, n_videos), "tag": rng.choice(["3,4", "5", None], n_videos),
                           "video_duration": rng.choice([5_000, 20_000, 90_000], n_videos),
                           "server_width": 720, "server_height": rng.choice([1280, 720], n_videos)})
    train = pd.DataFrame({"user_id": rng.integers(0, n_users, n_rows), "video_id": rng.integers(0, n_videos, n_rows),
                          "date": rng.integers(20220408, 20220422, n_rows), "hourmin": rng.integers(0, 24, n_rows) * 100,
                          "time_ms": np.arange(n_rows), "tab": rng.integers(0, 3, n_rows),
                          "is_click": rng.integers(0, 2, n_rows), "long_view": rng.integers(0, 2, n_rows), "is_like": 0})
    space = F.fit(train, users, videos)
    F.build(train, space, users, videos, is_train=True)  # sets the numeric standardisation
    return F, space, users, videos


def test_reward_broadcast_equals_feature_build():
    from receval.train import lgb_matrix
    from rlrec.rewards import Broadcast
    F, space, users, videos = _feature_world()
    bc = Broadcast(space, users, videos, date=20220422, hour=13)
    ids = np.array([4, 0, 2])
    grid = pd.DataFrame({"user_id": np.repeat(ids, len(bc.video_ids)), "video_id": np.tile(bc.video_ids, len(ids)),
                         "date": 20220422, "hourmin": 1300, "tab": 1, "time_ms": 0})
    direct = lgb_matrix(F.build(grid, space, users, videos, is_train=False))
    assert np.array_equal(bc.matrix(ids), direct)


def test_calibrated_tables_match_receval():
    from receval import calibrate as Cal
    from rlrec.rewards import calibrated, fit_calibrations
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 4000)
    group = rng.integers(0, 5, 4000)
    y = (rng.random(4000) < p * (0.6 + 0.1 * group)).astype(float)
    fitted = fit_calibrations(p, y, group)
    r1 = rng.uniform(0.01, 0.99, (3, 10)).astype(np.float32)
    item_group = np.arange(10) % 5
    r2 = calibrated("platt", r1, fitted, item_group)
    r3 = calibrated("grouped", r1, fitted, item_group)
    assert np.allclose(r2, Cal.apply_affine(fitted["platt"], r1), atol=1e-6)
    want = np.stack([Cal.apply_grouped(fitted["grouped"], r1[u], item_group) for u in range(3)])
    assert np.allclose(r3, want, atol=1e-6)


def test_video_text_rules():
    from rlrec.text import describe
    base = {f"{lvl}_level_category_name": None for lvl in ("first", "second", "third", "fourth")}
    good = {**base, "malformed": False, "caption": " 好看 #舞蹈 ", "show_cover_text": "封面",
            "first_level_category_name": "舞蹈", "second_level_category_name": "UNKNOWN", "third_level_category_name": "街舞"}
    assert describe(good) == "标题：好看 #舞蹈；封面文字：封面；类目：舞蹈 / 街舞"
    broken = {**good, "malformed": True}
    assert describe(broken) == "类目：舞蹈 / 街舞"
    empty_caption = {**good, "caption": "  ", "show_cover_text": "UNKNOWN"}
    assert describe(empty_caption) == "类目：舞蹈 / 街舞"
    assert describe({**base, "malformed": False, "caption": None, "show_cover_text": None}) == "无文本"


def test_list_counts_match_brute_force():
    from rlrec.data import Population
    from rlrec.evaluate import counts, row_ranks, value
    rng = np.random.default_rng(0)
    users, items, rows = 7, 30, 120
    scores = rng.normal(size=(users, items)).round(1)  # ties on purpose
    scores[rng.random((users, items)) < 0.2] = -np.inf
    pos = rng.integers(0, users, rows)
    item = rng.integers(0, items, rows)
    keep = np.isfinite(scores[pos, item])
    pos, item = pos[keep], item[keep]
    click = rng.integers(0, 2, len(pos)).astype(float)
    pop = Population(np.arange(users), pos, item, click, 1 - click, {})
    ranks = row_ranks(scores, pop, chunk=3)
    for r, (u, i) in enumerate(zip(pos, item)):
        better = (scores[u] > scores[u, i]) | ((scores[u] == scores[u, i]) & (np.arange(items) < i))
        assert ranks[r] == better.sum()
    c = counts(ranks, pop, (5,))[5]
    inside = ranks < 5
    assert c["rows"].sum() == inside.sum() and c["clicks"].sum() == (inside * click).sum()
    assert value(c) == pytest.approx((inside * click).sum() / inside.sum())


def _rl_case():
    codes, model, hist, allowed = _tiny(seed=6, users=4)
    allowed[:] = True
    tree = Tree(codes)
    ref = Generator(codes, hist.shape[1], d=32, layers=2, heads=4, dropout=0.0).eval()
    ref.load_state_dict(model.state_dict())
    logp0 = policy.exact_logprobs(ref, codes, tree, hist, allowed, CPU)
    rewards = np.tile(np.linspace(0, 1, len(codes.codes)).astype(np.float32), (len(hist), 1))
    return codes, model, ref, tree, hist, allowed, logp0, rewards, rl.RLData(hist, allowed, rewards, hist, allowed, logp0)


def test_rl_training_moves_towards_the_reward_and_records_levels():
    from rlrec.rl import train
    codes, model, ref, tree, hist, allowed, logp0, rewards, data = _rl_case()
    before = float((np.exp(logp0) * rewards).sum(1).mean())
    out = train(model, ref, codes, tree, data, lam=0.01, lr=3e-3, seed=0, dev=CPU, levels=(0.05, 0.2),
                max_steps=60, check_every=10, fine_every=2, users_per_step=4, samples=8)
    after = policy.exact_logprobs(model, codes, tree, hist, allowed, CPU)
    assert float((np.exp(after) * rewards).sum(1).mean()) > before
    sel = out["selected"]
    assert set(sel) == {0.05, 0.2} and all(s["state"] is not None for s in sel.values())
    reached = [s for s in sel.values() if s["reached"]]
    assert all(s["kl"] >= lvl for lvl, s in sel.items() if s["reached"]) and reached
    steps = [row["step"] for row in out["log"]]
    assert steps == sorted(steps) and steps[0] == 10
    gaps = [nxt["step"] - prev["step"] for prev, nxt in zip(out["log"], out["log"][1:])]
    assert not gaps or (set(gaps[:-1]) <= {2, 10} and 0 < gaps[-1] <= 10)  # the last may be cut by max_steps
    assert all(s_["aligned"] == (s_["reached"] and s_["kl"] <= 1.05 * lv) for lv, s_ in sel.items())


def test_rl_checks_every_step_from_the_first_until_the_level_is_passed():
    codes, model, ref, tree, _, _, _, _, data = _rl_case()
    out = rl.train(model, ref, codes, tree, data, lam=0.01, lr=3e-3, seed=0, dev=CPU, levels=(0.05, 5.0), max_steps=45,
                   check_every=10, fine_every=2, users_per_step=4, samples=8, every_step_until=0.05)
    passed = out["selected"][0.05]
    assert passed["reached"]
    steps = [row["step"] for row in out["log"]]
    assert steps[:passed["step"]] == list(range(1, passed["step"] + 1))
    assert set(np.diff(steps[passed["step"] - 1:-1])) == {10}  # back to coarse; the last may be cut by max_steps


def test_exact_logprobs_without_any_shared_three_code_prefix():
    codes, model, hist, allowed = _tiny(seed=7, items=30, levels=(4, 3, 3, 1))
    tree = Tree(codes)
    assert len(tree.multi) == 0
    exact = policy.exact_logprobs(model, codes, tree, hist, allowed, CPU, chunk=2)
    brute = _brute(model, codes, hist, allowed)
    assert np.allclose(exact[allowed], brute[allowed], atol=1e-5)
    assert np.allclose(np.log(np.exp(exact.astype(np.float64)).sum(1)), 0.0, atol=1e-5)


def test_fine_checks_persist_until_the_level_is_passed():
    from rlrec.rl import schedule
    levels, fine = (0.5, 1.0, 2.0), None
    nxt, fine = schedule(100, 0.30, list(levels), fine)  # far: coarse
    assert (nxt, fine) == (200, None)
    nxt, fine = schedule(200, 0.43, list(levels), fine)  # 86% of 0.5: fine
    assert (nxt, fine) == (210, 0.5)
    nxt, fine = schedule(210, 0.40, list(levels), fine)  # falls back below 85%: still fine
    assert (nxt, fine) == (220, 0.5)
    nxt, fine = schedule(220, 0.55, [1.0, 2.0], fine)  # passed 0.5, 1.0 not close: coarse again
    assert (nxt, fine) == (320, None)
    nxt, fine = schedule(320, 0.90, [1.0, 2.0], fine)  # close to 1.0
    assert (nxt, fine) == (330, 1.0)


def test_every_step_until_the_main_level_is_passed():
    from rlrec.rl import schedule
    nxt, fine = schedule(5, 0.20, [0.5, 1.0, 2.0], None, every_step_until=1.0)
    assert (nxt, fine) == (6, None)
    nxt, fine = schedule(6, 0.90, [1.0, 2.0], None, every_step_until=1.0)
    assert (nxt, fine) == (7, 1.0)
    nxt, fine = schedule(7, 1.01, [2.0], fine, every_step_until=1.0)  # passed: back to §6.2
    assert (nxt, fine) == (107, None)
    nxt, fine = schedule(107, 1.72, [2.0], fine, every_step_until=1.0)
    assert (nxt, fine) == (117, 2.0)


def test_user_batches_visit_every_user_once_per_round():
    gen = rl.user_batches(10, 4, np.random.default_rng(0))
    drawn = np.concatenate([next(gen) for _ in range(5)])  # exactly two rounds
    assert (np.bincount(drawn, minlength=10) == 2).all()


def test_max_rise_reads_the_fastest_hundred_steps():
    log = [{"step": 100, "subset_kl": 0.05}, {"step": 200, "subset_kl": 0.35}, {"step": 210, "subset_kl": 0.4}]
    assert rl.max_rise(log) == pytest.approx(0.4 - 0.08)  # from step 110 (0.08) to step 210


def test_max_slope_is_the_fastest_stretch_between_checks():
    log = [{"step": 100, "subset_kl": 0.1}, {"step": 200, "subset_kl": 0.85}, {"step": 210, "subset_kl": 1.1}]
    assert rl.max_slope(log) == pytest.approx(0.025)


def test_family_and_runs_follow_the_amendment():
    from rlrec import report
    from rlrec import stages as S
    assert list(S.HYPOTHESES) == list(report.HYPOTHESES) == ["H1", "H2", "H3", "H4", "H5", "H6"]
    assert len(S.MAIN_RL) == 36 and {r for r, *_ in S.MAIN_RL} == {"R1", "R3", "R4", "R5"}
    assert len(S.SECONDARY_RL) == 6 and all(grpo for *_, grpo in S.SECONDARY_RL)
    assert not [p_ for p_ in report.policy_list() if p_[0] == "O4" and p_[3] != "main"]
    assert len(report.policy_list()) == 55


def test_mixture_support_uses_the_averaged_policy():
    from rlrec import report
    prob_sum = np.array([[0.9 + 0.1, 0.1 + 0.9]], dtype=np.float32)  # two models, [.9, .1] and [.1, .9]
    assert report.mixture_support(prob_sum, 2) == pytest.approx([2.0])


def _fake_report():
    from rlrec import report
    rng = np.random.default_rng(0)
    names = sorted({x for a, b, _ in report.HYPOTHESES.values() for x in (a, b)})
    pol = {}
    for i, name in enumerate(names):
        rows = rng.poisson(5, 400).astype(float)
        clicks = rng.binomial(rows.astype(int), 0.2 + 0.05 * i).astype(float)
        counts = {"rows": rows, "clicks": clicks}
        pol[name] = {"mix": {500: {**counts, "long_view": clicks / 2}}, "per_seed": {s: {500: counts} for s in (0, 1, 2)}}
    return {"policies": pol}


def test_untestable_hypothesis_keeps_its_description_and_holds_its_place_in_holm():
    from receval import stats
    from rlrec import report
    family = list(report.HYPOTHESES)
    status = {h: {"testable": h != "H5", "reasons": ["checkpoint not aligned"] if h == "H5" else []} for h in family}
    st = report.statistics(_fake_report(), 500, family, status)["hypotheses"]
    assert "point" in st["H5"] and "ci95" in st["H5"] and "p" not in st["H5"] and st["H5"]["reasons"]
    assert st["H5"]["holds"] is False and st["H5"]["holm_reject"] is False
    expected = stats.holm([st[h].get("p", 1.0) for h in family])
    assert [st[h]["holm_reject"] for h in family] == [bool(x) for x in expected]


def test_descriptive_statistics_have_no_p_holm_or_verdict():
    from rlrec import report
    family = list(report.HYPOTHESES)
    status = {h: {"testable": True, "reasons": []} for h in family}
    out = report.statistics(_fake_report(), 500, family, status, descriptive=True)
    for row in out["hypotheses"].values():
        assert "point" in row and not {"p", "holm_reject", "holds", "mde"} & set(row)
    assert all("p" not in v["500"] for v in out["values"].values())


def test_tables_script_prints_every_state_of_a_hypothesis():
    import importlib.util
    from pathlib import Path

    from rlrec import report
    path = Path(__file__).resolve().parents[1] / "scripts" / "make_rlrec_tables.py"
    spec = importlib.util.spec_from_file_location("make_rlrec_tables", path)
    tables = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tables)
    family = list(report.HYPOTHESES)
    status = {h: {"testable": h != "H6", "reasons": ["x"] if h == "H6" else []} for h in family}
    st = report.statistics(_fake_report(), 500, family, status)
    text = "\n".join(tables.hypotheses(st, {}))
    assert "H6 随机数据" in text and "不可检验" in text
    hold = report.statistics(_fake_report(), 500, family, status, descriptive=True)
    assert "（只描述）" in "\n".join(tables.hypotheses(hold, {}, descriptive=True))


def test_test_evaluation_is_claimed_once_after_the_dry_run(tmp_path, monkeypatch):
    import json

    from rlrec import stages as S
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(S, "MANIFEST", manifest)
    monkeypatch.setattr(S, "TEST_LOG", tmp_path / "log.jsonl")
    monkeypatch.setattr(S, "_git", lambda *a: "head")
    monkeypatch.setattr(S.registry, "read", lambda runs: [])
    with pytest.raises(PermissionError, match="hold-report"):
        S._claim_test("t", None)
    dry_run = {"kind": "rlrec-report-hold", "status": "ok", "config": {"manifest_sha256": S.file_digest(manifest)}}
    monkeypatch.setattr(S.registry, "read", lambda runs: [dry_run])
    S._claim_test("t", None)
    with pytest.raises(PermissionError, match="did not finish"):
        S._claim_test("t", None)
    S._claim_test("t", "crashed before any output")
    with open(S.TEST_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"status": "completed"}) + "\n")
    with pytest.raises(PermissionError, match="already evaluated"):
        S._claim_test("t", "again")
    S.TEST_LOG.unlink()
    S._claim_test("t", None)
    manifest.write_text("{ }", encoding="utf-8")
    dry_run["config"]["manifest_sha256"] = S.file_digest(manifest)
    with pytest.raises(PermissionError, match="another manifest"):
        S._claim_test("t", "crashed")
    finished = {"kind": "rlrec-report-test", "status": "ok"}
    monkeypatch.setattr(S.registry, "read", lambda runs: [dry_run, finished])
    with pytest.raises(PermissionError, match="already evaluated"):
        S._claim_test("t", "the completion line was lost")


def _fake_chain(monkeypatch, data_of_sft="d"):
    from rlrec import stages as S
    inputs = {s: {"x": s} for s in S.CHAIN}
    stage_of = {tuple(sorted(v)): k for k, v in S.CHAIN.items()}
    runs = ["text", "reward_models", *S.CHAIN, "sft0"]
    started = [{"run_id": r, "status": "started", "data_hash": data_of_sft if r == "sft0" else "d",
                "config": {"inputs": inputs.get(r)}} for r in runs]
    ok = [{"run_id": r, "status": "ok", "config": {"selected": True}} for r in runs]  # a final record may lack the inputs
    monkeypatch.setattr(S.registry, "read", lambda runs_dir: started + ok)
    monkeypatch.setattr(S.registry, "data_hash", lambda: "d")
    monkeypatch.setattr(S, "latest", lambda s: {"run_id": s, "seeds": {"0": {"run_id": "sft0"}}})
    monkeypatch.setattr(S, "inputs", lambda *up: inputs[stage_of[tuple(sorted(up))]])
    monkeypatch.setattr(S, "rl_index", lambda which: {})
    return S


def test_chain_check_reads_the_start_records(monkeypatch):
    _fake_chain(monkeypatch)._check_chain()


def test_chain_check_stops_on_other_data(monkeypatch):
    with pytest.raises(PermissionError, match="other data"):
        _fake_chain(monkeypatch, data_of_sft="old")._check_chain()


def test_rl_runs_are_tied_to_their_upstream_stages(monkeypatch):
    from rlrec import stages as S
    monkeypatch.setattr(S, "inputs", lambda *s: {"sft": "new"})
    S._check_rl_inputs("run", {"inputs": {"sft": "new"}})
    with pytest.raises(PermissionError):
        S._check_rl_inputs("run", {"inputs": {"sft": "old"}})
