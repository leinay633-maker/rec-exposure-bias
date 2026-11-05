import json
import subprocess
from types import SimpleNamespace

import pandas as pd
import pytest

from receval import config, data
from receval.audit import eligible_users


def _log(dates):
    rows = []
    for i, d in enumerate(dates):
        rows.append({"user_id": i % 3, "video_id": i, "date": d, "hourmin": 100, "time_ms": 0,
                     "duration_ms": 1000, "is_rand": 0, "tab": 1, **{c: i % 2 for c in data.FEEDBACK}})
    return pd.DataFrame(rows)


def test_split_rejects_rows_outside_the_windows():
    frame = _log([20220422, 20220430, 20220510])
    with pytest.raises(ValueError, match="outside"):
        data._split(frame, "late", {"valid": config.VALID, "test": config.TEST})


def test_split_assigns_each_row_once():
    parts = data._split(_log([20220422, 20220428, 20220429, 20220508]), "late",
                        {"valid": config.VALID, "test": config.TEST})
    assert len(parts["valid"]) == 2 and len(parts["test"]) == 2


def test_sealed_rows_are_never_parsed_with_feedback(tmp_path, monkeypatch):
    monkeypatch.setitem(config.LOG_FILES, "standard_late", "late.csv")
    _log([20220422, 20220429, 20220430, 20220423, 20220501]).to_csv(tmp_path / "late.csv", index=False)
    seen = []
    real = pd.read_csv

    def spy(*args, **kwargs):
        frame = real(*args, **kwargs)
        seen.append(frame)
        return frame
    monkeypatch.setattr(data.pd, "read_csv", spy)
    parts = data._read_log("standard_late", tmp_path, {"valid": config.VALID, "test": config.TEST},
                           sealed=("test",))
    assert set(parts["test"].columns) == set(data.META) and len(parts["test"]) == 3
    assert len(parts["valid"]) == 2 and "is_click" in parts["valid"].columns
    for frame in seen:
        if "is_click" in frame.columns:
            assert not frame["date"].between(*config.TEST).any()


def test_unknown_cache_names_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown"):
        data.load("./random_test", cache=tmp_path, runs=tmp_path)


def _repo(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)

    def git(*args):
        return subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True).stdout
    git("init", "-b", "main")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    git("config", "core.autocrlf", "false")
    return git


def _commit(git, message):
    git("add", "-A")
    git("commit", "-m", message)
    return git("rev-parse", "HEAD").strip()


def _data(tmp_path):
    """A cache and raw logs whose fingerprints the data manifest records."""
    cache, raw = tmp_path / "cache", tmp_path / "raw"
    cache.mkdir()
    raw.mkdir()
    for fname in config.LOG_FILES.values():
        (raw / fname).write_text("x")
    pd.DataFrame({"a": [1]}).to_parquet(cache / "users.parquet")
    (cache / "manifest.json").write_text(json.dumps({
        "files": {"users": {"rows": 1, "sha256": data.file_digest(cache / "users.parquet")}},
        "raw": {k: data.file_digest(raw / f) for k, f in config.LOG_FILES.items()}}))
    return cache, raw


def _frozen_repo(tmp_path):
    git = _repo(tmp_path)
    cache, raw = _data(tmp_path)
    (tmp_path / ".gitignore").write_text("cache/\nraw/\nart/\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "PROTOCOL.md").write_text("v1\n")
    protocol = _commit(git, "protocol")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "m.py").write_text("x = 1\n")
    (tmp_path / "art").mkdir()
    artefacts, selection = [], {}
    for model, labels in data.REQUIRED_ARTEFACTS.items():
        for label, seeds in labels.items():
            for seed in seeds:
                path = tmp_path / "art" / f"{model}-{label}-{seed}.bin"
                path.write_bytes(f"{model}{label}{seed}".encode())
                artefacts.append({"model": model, "label": label, "seed": seed, "run_id": path.stem,
                                  "path": f"art/{path.name}", "sha256": data.file_digest(path)})
                selection.setdefault(model, {}).setdefault(label, {})[str(seed)] = path.stem
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "zoo_manifest.json").write_text(json.dumps({
        "data_manifest_sha256": data.file_digest(cache / "manifest.json"), "selection": selection,
        "artefacts": artefacts}))
    zoo = _commit(git, "zoo")
    (tmp_path / "configs" / "freeze.json").write_text(json.dumps({"protocol_commit": protocol, "zoo_commit": zoo}))
    _commit(git, "freeze")
    return git, cache, raw, tmp_path / "configs" / "freeze.json"


def _check(tmp_path, freeze, cache, raw):
    return data.check_freeze(freeze, repo=tmp_path, cache=cache, raw_dir=raw)


def test_freeze_chain_passes_when_everything_matches_and_protocol_only_grows(tmp_path):
    git, cache, raw, freeze = _frozen_repo(tmp_path)
    assert _check(tmp_path, freeze, cache, raw)["zoo_commit"]
    (tmp_path / "docs" / "PROTOCOL.md").write_text("v1\n## appended\n")
    _commit(git, "append")
    assert _check(tmp_path, freeze, cache, raw)["zoo_commit"]


def test_freeze_rejects_an_edited_protocol(tmp_path):
    git, cache, raw, freeze = _frozen_repo(tmp_path)
    (tmp_path / "docs" / "PROTOCOL.md").write_text("v1 edited\n")
    _commit(git, "edit")
    with pytest.raises(PermissionError, match="protocol text was edited"):
        _check(tmp_path, freeze, cache, raw)


def test_freeze_rejects_changed_artefacts_cache_and_raw_logs(tmp_path):
    _, cache, raw, freeze = _frozen_repo(tmp_path)
    (tmp_path / "art" / "dcn-is_click-1.bin").write_bytes(b"retrained")
    with pytest.raises(PermissionError, match="artefact"):
        _check(tmp_path, freeze, cache, raw)
    _, cache, raw, freeze = _frozen_repo(tmp_path / "b")
    pd.DataFrame({"a": [2]}).to_parquet(cache / "users.parquet")
    with pytest.raises(PermissionError, match="cache file"):
        _check(tmp_path / "b", freeze, cache, raw)
    _, cache, raw, freeze = _frozen_repo(tmp_path / "c")
    (raw / config.LOG_FILES["random"]).write_text("y")
    with pytest.raises(PermissionError, match="raw log"):
        _check(tmp_path / "c", freeze, cache, raw)


def test_freeze_rejects_a_missing_artefact(tmp_path):
    git, cache, raw, freeze = _frozen_repo(tmp_path)
    manifest_path = tmp_path / "configs" / "zoo_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artefacts"] = [a for a in manifest["artefacts"] if a["model"] != "din"]
    manifest_path.write_text(json.dumps(manifest))
    zoo = _commit(git, "zoo without din")
    freeze_data = json.loads(freeze.read_text())
    freeze.write_text(json.dumps({**freeze_data, "zoo_commit": zoo}))
    _commit(git, "refreeze")
    with pytest.raises(PermissionError, match="lacks required artefacts"):
        _check(tmp_path, freeze, cache, raw)


def test_freeze_rejects_artefacts_that_do_not_match_the_selection(tmp_path):
    git, cache, raw, freeze = _frozen_repo(tmp_path)
    manifest_path = tmp_path / "configs" / "zoo_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["selection"]["dcn"]["is_click"]["1"] = "some-other-run"
    manifest_path.write_text(json.dumps(manifest))
    zoo = _commit(git, "zoo pointing at another run")
    freeze.write_text(json.dumps({**json.loads(freeze.read_text()), "zoo_commit": zoo}))
    _commit(git, "refreeze")
    with pytest.raises(PermissionError, match="do not match"):
        _check(tmp_path, freeze, cache, raw)


def test_freeze_rejects_code_changed_after_the_zoo_freeze(tmp_path):
    git, cache, raw, freeze = _frozen_repo(tmp_path)
    (tmp_path / "src" / "m.py").write_text("x = 2\n")
    _commit(git, "tweak model")
    with pytest.raises(PermissionError, match="frozen paths changed"):
        _check(tmp_path, freeze, cache, raw)


def test_freeze_rejects_a_dirty_tree_and_bad_chains(tmp_path):
    _, cache, raw, freeze = _frozen_repo(tmp_path)
    (tmp_path / "src" / "m.py").write_text("dirty\n")
    with pytest.raises(PermissionError, match="dirty"):
        _check(tmp_path, freeze, cache, raw)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"protocol_commit": "a" * 40, "zoo_commit": "b" * 40}))
    with pytest.raises(PermissionError, match="existing commits"):
        data.check_freeze(bad, repo=tmp_path, cache=cache, raw_dir=raw)
    zoo = json.loads(freeze.read_text())["zoo_commit"]
    bad.write_text(json.dumps({"protocol_commit": zoo[:10], "zoo_commit": zoo}))  # one commit, two spellings
    with pytest.raises(PermissionError, match="strict ancestor"):
        data.check_freeze(bad, repo=tmp_path, cache=cache, raw_dir=raw)
    with pytest.raises(PermissionError, match="missing"):
        data.check_freeze(tmp_path / "none.json", repo=tmp_path, cache=cache, raw_dir=raw)


def _selection(tmp_path, monkeypatch):
    """A complete fake zoo: registry entries, artefact files, and the selection written from them."""
    from receval import tune
    cache, raw = _data(tmp_path)
    monkeypatch.setattr(config, "CACHE", cache)
    (tmp_path / "art").mkdir()
    selection, entries = {}, []
    for model, labels in data.REQUIRED_ARTEFACTS.items():
        selection[model] = {"config": {"dim": 16}}
        for label, seeds in labels.items():
            selection[model][label] = {}
            for seed in seeds:
                run_id = f"{model}-{label}-{seed}"
                (tmp_path / "art" / run_id).write_bytes(run_id.encode())
                entries.append({"run_id": run_id, "model": model, "label": label, "seed": seed, "status": "ok",
                                "artefact": f"art/{run_id}", "config": {"dim": 16}, "info": {"valid_uauc": 0.6},
                                "sha256": data.file_digest(tmp_path / "art" / run_id)})
                selection[model][label][seed] = run_id
    monkeypatch.setattr(tune.registry, "read", lambda *args: entries)
    chosen = tune.write_selection(selection, out=tmp_path / "zoo" / "selection.json")
    return SimpleNamespace(tune=tune, cache=cache, raw=raw, selection=selection, entries=entries, chosen=chosen)


def _passed_runs(tmp_path, zoo, chosen, kinds=("analysis-proxy", "power-check")):
    """A registry file with the zoo's own runs plus successful gate runs bound to `chosen`."""
    from receval import registry
    runs = tmp_path / "runs"
    runs.mkdir(exist_ok=True)
    bound = {"selection_sha256": data.file_digest(chosen), "data_hash": registry.data_hash(zoo.cache),
             "code_hash": registry.code_hash(tmp_path)}
    lines = zoo.entries + [{"kind": k, "status": "ok", **bound} for k in kinds]
    (runs / "registry.jsonl").write_text("".join(json.dumps(e) + "\n" for e in lines))
    return runs


def _freeze_zoo(tmp_path, zoo, chosen, runs):
    return zoo.tune.write_zoo_manifest(chosen, repo=tmp_path, cache=zoo.cache, runs=runs, raw_dir=zoo.raw)


def test_zoo_manifest_from_the_selection_binds_every_required_artefact(tmp_path, monkeypatch):
    zoo = _selection(tmp_path, monkeypatch)
    monkeypatch.undo()  # the gates must read the real registry file written below
    monkeypatch.setattr(config, "CACHE", zoo.cache)
    manifest = json.loads(_freeze_zoo(tmp_path, zoo, zoo.chosen, _passed_runs(tmp_path, zoo, zoo.chosen)).read_text())
    data.verify_manifest(manifest, tmp_path)
    assert manifest["data_manifest"]["files"]["users"]["rows"] == 1
    assert manifest["selection"]["din"]["long_view"]["2"] == "din-long_view-2"
    entry = next(a for a in manifest["artefacts"] if a["run_id"] == "din-long_view-2")
    assert entry["config"] == {"dim": 16} and entry["valid_metric"] == 0.6


def test_zoo_manifest_refuses_an_incomplete_zoo_missing_dry_runs_and_changed_cache_files(tmp_path, monkeypatch):
    zoo = _selection(tmp_path, monkeypatch)
    del zoo.selection["din"]["long_view"]
    incomplete = zoo.tune.write_selection(zoo.selection, out=tmp_path / "zoo" / "incomplete.json")
    monkeypatch.undo()
    monkeypatch.setattr(config, "CACHE", zoo.cache)
    with pytest.raises(PermissionError, match="lacks required artefacts"):
        _freeze_zoo(tmp_path, zoo, incomplete, _passed_runs(tmp_path, zoo, incomplete))
    with pytest.raises(PermissionError, match="power-check"):
        _freeze_zoo(tmp_path, zoo, zoo.chosen, _passed_runs(tmp_path, zoo, zoo.chosen, kinds=("analysis-proxy",)))
    runs = _passed_runs(tmp_path, zoo, zoo.chosen)
    users = (zoo.cache / "users.parquet").read_bytes()
    (zoo.cache / "users.parquet").write_bytes(b"rewritten")  # content changed, manifest untouched
    with pytest.raises(PermissionError, match="cache file users"):
        _freeze_zoo(tmp_path, zoo, zoo.chosen, runs)
    (zoo.cache / "users.parquet").write_bytes(users)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "late_edit.py").write_text("x = 1\n")  # code changed after the dry runs
    with pytest.raises(PermissionError, match="analysis-proxy"):
        _freeze_zoo(tmp_path, zoo, zoo.chosen, runs)


def test_one_run_cannot_stand_in_for_another_seed(tmp_path, monkeypatch):
    zoo = _selection(tmp_path, monkeypatch)
    relabelled = json.loads(json.dumps(zoo.selection))
    relabelled["dcn"]["is_click"]["2"] = "dcn-is_click-1"  # seed 1's run listed again as seed 2
    with pytest.raises(PermissionError, match="does not match the registered run"):
        zoo.tune.write_selection(relabelled, out=tmp_path / "zoo" / "relabelled.json")
    # A hand-edited selection: a distinct file and run id, unique and unchanged, but registered as seed 1.
    (tmp_path / "art" / "extra").write_bytes(b"extra")
    extra = {"run_id": "extra", "artefact": "art/extra", "sha256": data.file_digest(tmp_path / "art" / "extra")}
    zoo.entries.append({**next(e for e in zoo.entries if e["run_id"] == "dcn-is_click-1"), **extra})
    payload = json.loads(zoo.chosen.read_text())
    next(a for a in payload["artefacts"] if a["run_id"] == "dcn-is_click-2").update(
        run_id="extra", path="art/extra", sha256=extra["sha256"])
    payload["selection"]["dcn"]["is_click"]["2"] = "extra"
    edited = tmp_path / "zoo" / "edited.json"
    edited.write_text(json.dumps(payload))
    monkeypatch.undo()
    monkeypatch.setattr(config, "CACHE", zoo.cache)
    data.verify_manifest(payload, tmp_path)  # the file-level checks alone let it through
    with pytest.raises(PermissionError, match="does not match the registered run"):
        _freeze_zoo(tmp_path, zoo, edited, _passed_runs(tmp_path, zoo, edited))


def test_manifest_rejects_one_run_listed_for_two_seeds(tmp_path):
    _frozen_repo(tmp_path)
    manifest = json.loads((tmp_path / "configs" / "zoo_manifest.json").read_text())
    artefact = {(a["model"], a["label"], a["seed"]): a for a in manifest["artefacts"]}
    one = artefact[("dcn", "is_click", 1)]
    artefact[("dcn", "is_click", 2)].update(run_id=one["run_id"], path=one["path"], sha256=one["sha256"])
    manifest["selection"]["dcn"]["is_click"]["2"] = one["run_id"]
    with pytest.raises(PermissionError, match="one run_id for more than one"):
        data.verify_manifest(manifest, tmp_path)


def test_sealed_load_needs_a_completed_opening_with_matching_hash(tmp_path):
    cache, runs = tmp_path / "cache", tmp_path / "runs"
    cache.mkdir()
    runs.mkdir()
    with pytest.raises(PermissionError, match="sealed"):
        data.load("random_test", cache=cache, runs=runs)
    pd.DataFrame({"a": [1]}).to_parquet(cache / "random_test.parquet")
    with pytest.raises(PermissionError, match="sealed"):
        data.load("random_test", cache=cache, runs=runs)
    record = {"status": "completed", "files": {"random_test": {"sha256": data.file_digest(cache / "random_test.parquet")}}}
    (runs / data.OPENINGS).write_text(json.dumps(record) + "\n")
    assert len(data.load("random_test", cache=cache, runs=runs)) == 1
    pd.DataFrame({"a": [2]}).to_parquet(cache / "random_test.parquet")
    with pytest.raises(PermissionError, match="modified"):
        data.load("random_test", cache=cache, runs=runs)


def test_open_test_refuses_a_second_opening_and_stray_files(tmp_path):
    cache, runs = tmp_path / "cache", tmp_path / "runs"
    cache.mkdir()
    runs.mkdir()
    pd.DataFrame({"a": [1]}).to_parquet(cache / "standard_test.parquet")
    with pytest.raises(PermissionError, match="stray"):
        data.open_test(cache=cache, runs=runs, freeze_file=tmp_path / "f.json", repo=tmp_path)
    (runs / data.OPENINGS).write_text(json.dumps({"status": "completed"}) + "\n")
    with pytest.raises(PermissionError, match="already opened"):
        data.open_test(cache=cache, runs=runs, freeze_file=tmp_path / "f.json", repo=tmp_path)


def test_eligible_users_need_both_classes():
    frame = pd.DataFrame({"user_id": [1, 1, 2, 2, 3], "is_click": [0, 1, 1, 1, 0]})
    assert eligible_users(frame, "is_click") == {1}
