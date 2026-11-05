import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from receval import metrics, stats


@pytest.fixture
def sample():
    rng = np.random.default_rng(0)
    users = rng.integers(0, 40, 2_000)
    y = rng.integers(0, 2, 2_000)
    score = np.round(rng.random(2_000) + 0.3 * y, 2)  # rounding creates ties
    return users, y, score


def test_auc_matches_sklearn_with_ties(sample):
    _, y, score = sample
    assert metrics.auc(y, score) == pytest.approx(roc_auc_score(y, score), abs=1e-12)


def test_per_user_auc_matches_a_loop(sample):
    users, y, score = sample
    table = metrics.per_user_auc(users, y, score)
    for user, row in table.iterrows():
        mask = users == user
        assert row["auc"] == pytest.approx(roc_auc_score(y[mask], score[mask]), abs=1e-12)
    both = [u for u in np.unique(users) if 0 < y[users == u].sum() < (users == u).sum()]
    assert sorted(table.index) == sorted(both)


def test_uauc_and_gauc_weights():
    table = pd.DataFrame({"auc": [1.0, 0.5], "rows": [2, 8], "pos": [1, 4], "neg": [1, 4]}, index=[7, 9])
    assert metrics.uauc(table) == pytest.approx(0.75)
    assert metrics.uauc(table, users=[9]) == pytest.approx(0.5)
    assert metrics.gauc(table) == pytest.approx(0.6)


def test_logloss_and_brier_match_sklearn(sample):
    _, y, score = sample
    p = np.clip(score / score.max(), 0.01, 0.99)
    assert metrics.logloss(y, p) == pytest.approx(log_loss(y, p), rel=1e-9)
    assert metrics.brier(y, p) == pytest.approx(brier_score_loss(y, p), rel=1e-12)


def test_ece_is_zero_for_perfectly_calibrated_bins():
    y = np.array([0, 1] * 50)
    p = np.full(100, 0.5)
    assert metrics.ece(y, p, bins=5) == pytest.approx(0.0)


def test_holm_step_down():
    assert stats.holm([0.01, 0.04, 0.03, 0.2], alpha=0.05).tolist() == [True, False, False, False]
    assert stats.holm([0.001, 0.012, 0.02], alpha=0.05).tolist() == [True, True, True]


def test_two_sided_p():
    assert stats.two_sided_p([1, 2, 3, -1]) == pytest.approx(0.5)
    assert stats.two_sided_p([1, 2, 3]) == 0.0


def test_joint_bootstrap_is_reproducible_and_shares_resamples():
    per_user = np.random.default_rng(1).random((50, 3, 2))
    per_user[:, :, 1] = per_user[:, :, 0]  # identical sets must give identical replicates
    a = stats.joint_user_bootstrap(per_user, samples=200, seed=5)
    b = stats.joint_user_bootstrap(per_user, samples=200, seed=5)
    assert a.shape == (200, 3, 2)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(a[:, :, 0], a[:, :, 1])
    assert abs(a[:, 0, 0].mean() - per_user[:, 0, 0].mean()) < 0.02


def test_pairwise_flips_detects_an_opposite_significant_pair():
    rng = np.random.default_rng(2)
    users = 400
    per_user = np.empty((users, 2, 2))
    per_user[:, 0, 0] = 0.70 + 0.01 * rng.standard_normal(users)
    per_user[:, 1, 0] = 0.60 + 0.01 * rng.standard_normal(users)
    per_user[:, 0, 1] = 0.55 + 0.01 * rng.standard_normal(users)
    per_user[:, 1, 1] = 0.65 + 0.01 * rng.standard_normal(users)
    point = per_user.mean(axis=0)
    reps = stats.joint_user_bootstrap(per_user, samples=500, seed=0)
    (row,) = stats.pairwise_flips(point, reps)
    assert row["flip"] and row["significant"] == [True, True]
    assert stats.kendall([0.7, 0.6, 0.5], [0.5, 0.6, 0.7]) == pytest.approx(-1.0)
