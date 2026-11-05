import pytest

from receval import config, registry


def test_logged_runs_leave_a_start_and_an_outcome(tmp_path):
    with registry.logged("unit", {"a": 1}, runs=tmp_path) as run:
        run["output"] = "x"
    with pytest.raises(ValueError):
        with registry.logged("unit", runs=tmp_path):
            raise ValueError("boom")
    entries = registry.read(tmp_path)
    assert [e["status"] for e in entries] == ["started", "ok", "started", "failed"]
    assert entries[0]["run_id"] == entries[1]["run_id"] and entries[1]["output"] == "x"
    assert "boom" in entries[3]["error"] and entries[0]["code_hash"] and entries[0]["config_hash"]
    assert registry.succeeded("unit", tmp_path, output="x") and not registry.succeeded("unit", tmp_path, output="y")


def test_code_hash_ignores_line_endings_but_not_content(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "m.py").write_bytes(b"x = 1\n")
    lf = registry.code_hash(tmp_path)
    (tmp_path / "src" / "m.py").write_bytes(b"x = 1\r\n")
    assert registry.code_hash(tmp_path) == lf
    (tmp_path / "src" / "m.py").write_bytes(b"x = 2\n")
    assert registry.code_hash(tmp_path) != lf


def test_a_failed_run_keeps_what_the_caller_recorded_and_marks_a_missing_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE", tmp_path / "no-cache")
    with pytest.raises(FileNotFoundError):
        with registry.logged("unit", runs=tmp_path) as run:
            run["output"] = "out-dir"
            raise FileNotFoundError("selection.json")
    started, failed = registry.read(tmp_path)
    assert started["data_hash"].startswith("unavailable") and failed["output"] == "out-dir"


def test_a_missing_selection_still_leaves_a_start_and_a_failure(monkeypatch):
    from receval import orchestrate
    entries = []
    monkeypatch.setattr(registry, "record", lambda entry, runs=None: entries.append(entry))
    monkeypatch.setattr(orchestrate, "SELECTION", config.ROOT / "no-such-dir" / "selection.json")
    with pytest.raises(FileNotFoundError):
        orchestrate.run_power_check()
    assert [e["status"] for e in entries] == ["started", "failed"]
    assert entries[1]["source"] == "no-such-dir/selection.json" and "output" in entries[1]


def test_gate_runs_fail_without_estimates_or_users():
    import math

    from receval.orchestrate import require_estimates, require_population
    good = {"users": 50, **{k: {"median": 0.01} for k in ("se_standard", "se_random", "se_did")}}
    require_estimates(good, "ranking")
    require_estimates({**good, "h3_mde_did": {"H3a": 0.016, "H3b": 0.017}}, "recall")
    for bad in ({"users": 0, "status": "fewer than two users"}, {**good, "se_did": {"median": math.nan}},
                {**good, "h3_mde_did": {"H3a": math.nan}}, {}):
        with pytest.raises(RuntimeError):
            require_estimates(bad, "x")
    require_population({"users": 3}, "ranking")
    with pytest.raises(RuntimeError):
        require_population({"users": 0, "status": "no common users; nothing to compare"}, "recall")
