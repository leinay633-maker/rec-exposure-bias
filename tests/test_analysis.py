import numpy as np
import pandas as pd
import pytest

from receval import analysis as A
from receval.features import LABELS


def _built(users, labels):
    lab = np.zeros((len(users), len(LABELS)), dtype=np.float32)
    lab[:, 0] = labels
    return {"user": np.asarray(users), "labels": lab}


def test_common_users_need_both_classes_in_every_set():
    a = _built([1, 1, 2, 2, 3, 3], [0, 1, 0, 1, 1, 1])
    b = _built([1, 1, 2, 3, 3], [1, 0, 1, 0, 1])
    assert A.common_users([a, b], "is_click") == [1]


def test_holm_first_step_multiplier():
    assert A.HOLM_FIRST_STEP == pytest.approx(3.88, abs=0.01)
    assert A.UNADJUSTED == pytest.approx(2.80, abs=0.01)
    assert A.stats.mde_multipliers(6)[0] == pytest.approx(3.48, abs=0.01)  # the six recall pairs


def _flip_cube(users=600, seeds=3, seed=0):
    """Model 0 beats model 1 on set 0 and loses on set 1, consistently across seeds."""
    rng = np.random.default_rng(seed)
    cube = np.empty((users, 3, 2, seeds))
    for k in range(seeds):
        cube[:, 0, 0, k] = 0.70 + 0.02 * rng.standard_normal(users)
        cube[:, 1, 0, k] = 0.60 + 0.02 * rng.standard_normal(users)
        cube[:, 0, 1, k] = 0.55 + 0.02 * rng.standard_normal(users)
        cube[:, 1, 1, k] = 0.65 + 0.02 * rng.standard_normal(users)
        cube[:, 2, :, k] = 0.62 + 0.02 * rng.standard_normal((users, 2))
    return cube


def test_ranking_comparison_finds_a_seed_supported_flip():
    out = A.ranking_comparison(_flip_cube(), order=("a", "b", "c"), samples=500, seeded=("a",))
    pair = next(r for r in out["pairs"] if r["pair"] == ["a", "b"])
    assert pair["flip_final"] and pair["seed_support"] == 3
    assert out["flips"] >= 1 and -1 <= out["kendall_tau_b"] <= 1


def test_seed_rule_blocks_a_flip_that_one_seed_drives():
    cube = _flip_cube()
    cube[:, 0, 1, 1:] = 0.75  # two seeds no longer flip; the seed mean still might
    out = A.ranking_comparison(cube, order=("a", "b", "c"), samples=500, seeded=("a",))
    pair = next(r for r in out["pairs"] if r["pair"] == ["a", "b"])
    assert not pair["flip_final"]


def test_gap_changes_count_seed_supported_changes_and_name_the_direction():
    out = A.ranking_comparison(_flip_cube(), order=("a", "b", "c"), samples=500, seeded=("a",))
    changes = {tuple(r["pair"]): r for r in out["gap_changes"]["pairs"]}
    assert changes[("a", "b")]["change_final"] and changes[("a", "b")]["direction"] == "reversed"
    assert changes[("a", "b")]["did"] == pytest.approx(-0.20, abs=0.01)
    assert out["gap_changes"]["changes"] == 3


def test_seed_rule_blocks_a_gap_change_that_one_seed_drives():
    cube = _flip_cube()
    cube[:, 0, 1, 1:] = 0.80  # seeds 1 and 2 move the gap the other way; the seed mean still falls
    out = A.ranking_comparison(cube, order=("a", "b", "c"), samples=500, seeded=("a",))
    pair = next(r for r in out["gap_changes"]["pairs"] if r["pair"] == ["a", "b"])
    assert pair["did"] < 0 and pair["significant"] and pair["seed_support"] == 1 and not pair["change_final"]


def test_gap_scaling_recovers_a_uniform_shrink():
    point = np.column_stack([np.linspace(0.6, 0.7, 5), 0.3 + 0.5 * np.linspace(0.6, 0.7, 5)])
    reps = point[None] + 1e-4 * np.random.default_rng(0).standard_normal((200, 5, 2))
    out = A.gap_scaling(point, reps)
    assert out["slope"] == pytest.approx(0.5) and out["ci"][0] < 0.5 < out["ci"][1]
    assert out["r2"] == pytest.approx(1.0) and out["undefined_replicates"] == 0


def test_gap_scaling_slope_alone_hides_pair_specific_changes():
    """Same slope 0.5 as a uniform shrink, but one pair reverses: only R^2 and the residuals show it."""
    point = np.array([[0.5, 0.5], [0.6, 0.8], [0.7, 0.6]])
    out = A.gap_scaling(point, point[None])
    assert out["slope"] == pytest.approx(0.5) and out["r2"] < 0.5


def test_gap_scaling_is_undefined_when_the_standard_set_cannot_separate_models():
    point = np.array([[0.6, 0.5], [0.6, 0.7], [0.6, 0.6]])
    out = A.gap_scaling(point, np.repeat(point[None], 50, axis=0))
    assert np.isnan(out["slope"]) and out["undefined_replicates"] == 50 and np.isnan(out["ci"][0])
    # Seven equal 0.1s: the rounded mean leaves a spread of about 1e-33, which must still count as a tie.
    tied = np.column_stack([np.full(7, 0.1), np.linspace(0.5, 0.6, 7)])
    out = A.gap_scaling(tied, np.repeat(tied[None], 4, axis=0))
    assert np.isnan(out["slope"]) and np.isnan(out["r2"]) and out["undefined_replicates"] == 4


def test_direction_labels_include_unchanged():
    assert A._direction([0.02, 0.02]) == "unchanged" and A._direction([0.02, -0.01]) == "reversed"
    assert A._direction([0.02, 0.03]) == "widened" and A._direction([-0.02, -0.01]) == "narrowed"


def test_empty_populations_give_an_explicit_status():
    from receval.orchestrate import _recall_se
    from receval.retrieval import evaluate_scorer
    assert A.ranking_comparison(np.empty((0, 3, 2, 1)), order=("a", "b", "c"))["users"] == 0
    empty = evaluate_scorer(None, [], None, {})  # no user to score: columns kept, nothing computed
    assert empty.empty and list(empty.columns) == ["targets", "recall@50", "recall@100", "recall@200", "ndcg@100"]
    tables = [{m: [empty] for m in A.RECALL_ORDER} for _ in range(2)]
    assert "status" in A.recall_comparison(tables)
    assert "status" in A.se_summary(np.empty((0, 4, 2)), A.RECALL_ORDER)
    assert _recall_se({}, None, [{1: np.array([0])}, {2: np.array([0])}])["users"] == 0


def test_mechanism_scores_each_seed_instead_of_averaging_predictions():
    from receval.orchestrate import mechanism
    y = np.array([0, 1, 0, 1], dtype=np.float32)
    built = _built([1, 1, 2, 2], y)
    built["video"] = np.array([10, 10, 10, 10])
    # Per-seed AUCs 0.75 and 1.0 average to 0.875; the averaged predictions would score 0.75.
    seeds = np.array([[0.1, 0.9, 0.8, 0.2], [0.2, 0.3, 0.1, 0.4]])
    out = mechanism(built, {"m": seeds}, {10: 0}, "is_click")["m"][0]
    assert out["auc"] == pytest.approx(np.mean([A.metrics.auc(y, s) for s in seeds]))
    assert out["auc"] != pytest.approx(A.metrics.auc(y, seeds.mean(axis=0)))


def _calibration_data(rows=4000, seed=0):
    """True logit z; the model reports sigmoid(0.5 + 2z), wrong in level and in slope."""
    rng = np.random.default_rng(seed)

    def part(n):
        z = rng.normal(-1.0, 1.0, n)
        y = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(np.float32)
        built = _built(rng.integers(0, 100, n), y)
        built["video"] = rng.integers(0, 50, n)
        return built, (1 / (1 + np.exp(-(0.5 + 2 * z))))[None, :]
    (calib, pc), (test, pt) = part(rows), part(rows)
    return calib, test, {"m": pc}, {"m": pt}, {v: v % 5 for v in range(50)}


def test_calibration_study_detects_a_slope_error_and_reports_secondary_metrics():
    calib, test, pc, pt, groups = _calibration_data()
    out = A.calibration_study(calib, test, pc, pt, groups, order=("m",), boot=200)
    cell = out["by_model_budget"]["m@all"]
    assert out["H2a"] and cell["intercept_minus_platt"]["point"] > 0
    assert cell["brier"]["platt"] < cell["brier"]["uncalibrated"] and np.isfinite(cell["ece"]["isotonic"])
    assert set(cell["uauc_random_test"]) == {"uncalibrated", "grouped"}
    assert "uauc_random_test" not in out["by_model_budget"]["m@1000"] and not out["fit_failures"]
    assert cell["logloss_user_mean"]["constant"] > cell["logloss_user_mean"]["platt"]
    low, high = out["by_model_budget"]["m@16000"]["spread_across_repeats"]["intercept_minus_platt"]
    assert low <= high


def test_a_failed_calibrator_fit_counts_as_not_holding(monkeypatch):
    calib, test, pc, pt, groups = _calibration_data()

    def fail(p, y):
        raise RuntimeError("calibrator fit failed: forced")
    monkeypatch.setattr(A.C, "fit_platt", fail)
    out = A.calibration_study(calib, test, pc, pt, groups, order=("m",), boot=200)
    assert not out["H2a"] and not out["H2b_overall"]
    assert {f["method"] for f in out["fit_failures"]} == {"platt", "grouped"}
    cell = out["by_model_budget"]["m@all"]
    assert np.isnan(cell["intercept_minus_platt"]["point"]) and np.isfinite(cell["logloss_user_mean"]["intercept"])


def test_recall_did_signs_follow_the_hypotheses():
    rng = np.random.default_rng(1)
    users = list(range(400))

    def frame(level):
        values = np.clip(level + 0.05 * rng.standard_normal(400), 0, 1)
        return pd.DataFrame({m: values for m in ("recall@50", "recall@100", "recall@200", "ndcg@100")}, index=users)
    standard = {"recall_popularity": [frame(0.5)], "recall_itemcf": [frame(0.3)],
                "recall_two_tower": [frame(0.2)] * 3, "recall_two_tower_logq": [frame(0.25)] * 3}
    random = {"recall_popularity": [frame(0.1)], "recall_itemcf": [frame(0.2)],
              "recall_two_tower": [frame(0.1)] * 3, "recall_two_tower_logq": [frame(0.3)] * 3}
    out = A.recall_comparison([standard, random], samples=500)
    assert out["did"]["H3a"]["point"] < 0 and out["did"]["H3a"]["holds"]
    assert out["did"]["H3b"]["point"] > 0 and out["did"]["H3b"]["holds"]
