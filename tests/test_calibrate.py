import numpy as np
import pytest

from receval import calibrate as C


def _synthetic(a, b, n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1.5, n)
    p_true = C.sigmoid(a + b * z)
    y = (rng.random(n) < p_true).astype(float)
    return C.sigmoid(z), y


def test_platt_recovers_known_parameters():
    p, y = _synthetic(-1.2, 0.6)
    a, b = C.fit_platt(p, y)
    assert a == pytest.approx(-1.2, abs=0.08) and b == pytest.approx(0.6, abs=0.05)


def test_platt_slope_is_never_negative():
    p, y = _synthetic(0.0, -1.0)
    assert C.fit_platt(p, y)[1] == pytest.approx(0.0, abs=1e-8)


def test_intercept_only_shifts_the_level():
    p, y = _synthetic(-1.5, 1.0)
    a, b = C.fit_intercept(p, y)
    assert b == 1.0 and a == pytest.approx(-1.5, abs=0.08)


def test_isotonic_clips_outside_the_fitted_range():
    model = C.fit_isotonic([0.2, 0.4, 0.6], [0, 1, 1])
    out = model.predict([0.0, 1.0])
    assert out[0] == pytest.approx(C.EPS) and out[1] == pytest.approx(1 - C.EPS)


def test_grouped_falls_back_to_global_for_degenerate_groups():
    p, y = _synthetic(-1.0, 0.8, n=6_000)
    group = np.zeros(len(p), int)
    group[:10] = 1
    y[:10] = 0  # group 1 has a single class
    fitted = C.fit_grouped(p, y, group, groups=3)
    assert fitted["groups"][1] == fitted["global"]
    assert fitted["groups"][2] == fitted["global"]  # empty group
    shrunk = fitted["groups"][0]
    assert shrunk != fitted["global"]
    out = C.apply_grouped(fitted, p, group)
    assert ((out > 0) & (out < 1)).all()


def test_popularity_groups_are_equal_sized_and_deterministic():
    counts = {1: 0, 2: 5, 3: 5, 4: 9, 5: 1, 6: 0}
    mapping = C.popularity_groups([1, 2, 3, 4, 5, 6], counts, groups=3)
    assert mapping == {1: 0, 6: 0, 5: 1, 2: 1, 3: 2, 4: 2}


def test_subsamples_are_nested_and_without_replacement():
    small, large = C.subsample(100, 10, repeat=3), C.subsample(100, 40, repeat=3)
    assert len(set(large)) == 40 and list(large[:10]) == list(small)
    assert len(C.subsample(100, None, repeat=0)) == 100
