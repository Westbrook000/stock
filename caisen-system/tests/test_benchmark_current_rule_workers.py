import json
from pathlib import Path


def test_matrix_schedule_runs_each_spawn_size_once_and_selects_minimum_within_five_percent():
    from scripts import benchmark_current_rule_workers as runner

    assert runner._candidate_worker_schedule(10, matrix=True) == (2, 4, 6, 8, 10)
    assert runner._candidate_worker_schedule(10, matrix=False) == (10,)
    metrics = {
        2: {"seconds": 12.0, "peak": 1024, "passed": True},
        4: {"seconds": 11.0, "peak": 1024, "passed": True},
        6: {"seconds": 9.8, "peak": 1024, "passed": True},
        8: {"seconds": 9.3, "peak": 1024, "passed": True},
        10: {"seconds": 9.0, "peak": 1024, "passed": True},
    }
    selected, fastest, eligible = runner._select_matrix_worker(
        metrics, serial_seconds=20.0, cap_bytes=2048)
    assert (selected, fastest, eligible) == (8, 9.0, [8, 10])


def test_matrix_selection_excludes_over_memory_and_any_failed_parity_or_accounting_gate():
    from scripts import benchmark_current_rule_workers as runner

    metrics = {
        2: {"seconds": 9.0, "peak": 2049, "passed": True},
        4: {"seconds": 9.2, "peak": 1024, "passed": True},
        6: {"seconds": 8.0, "peak": 1024, "passed": False},
    }
    selected, fastest, eligible = runner._select_matrix_worker(
        metrics, serial_seconds=20.0, cap_bytes=2048)
    assert (selected, fastest, eligible) == (4, 9.2, [4])
    assert not runner._matrix_passed(True, metrics, source_unchanged=True, selected=selected)
    assert not runner._matrix_passed(True, {4: {"passed": False}},
                                     source_unchanged=True, selected=None)


def test_recognition_profiles_are_explicitly_normalized():
    from scripts import benchmark_current_rule_workers as runner

    default_config, default_metadata = runner.resolve_recognition_profile("default")
    assert default_config is None
    assert default_metadata == {"name": "default", "cache_key": None,
                                "recognition_experiment": None}

    g1_config, g1_metadata = runner.resolve_recognition_profile("g1")
    assert g1_config.experiment_id == "internal_rebound_coverage_g1_v1"
    assert g1_config.require_internal_rebound_coverage is True
    assert g1_metadata["name"] == "g1"
    assert g1_metadata["cache_key"] == list(g1_config.cache_key())
    assert g1_metadata["recognition_experiment"]["experiment_id"] == g1_config.experiment_id
    assert g1_metadata["recognition_experiment"]["cache_key"] == list(g1_config.cache_key())

    try:
        runner.resolve_recognition_profile("unknown")
    except ValueError as exc:
        assert "recognition profile" in str(exc)
    else:
        raise AssertionError("unknown recognition profiles must be rejected")


def test_cli_parses_g1_profile_and_separate_certificate_path():
    from scripts import benchmark_current_rule_workers as runner

    args = runner._build_parser().parse_args([
        "--recognition-profile", "g1", "--certificate-output", "/tmp/e1-worker-certificate.json"])
    assert args.recognition_profile == "g1"
    assert str(args.certificate_output) == "/tmp/e1-worker-certificate.json"


def test_g1_profile_requires_independent_certificate_path(tmp_path):
    from scripts import benchmark_current_rule_workers as runner

    try:
        runner.run_policy("missing-reference", "missing-snapshot", "missing-universe",
                          tmp_path / "run", recognition_profile="g1")
    except ValueError as exc:
        assert "certificate_output" in str(exc)
    else:
        raise AssertionError("G1 certificates must use an explicit independent path")
    assert not (tmp_path / "run").exists()


def test_policy_runner_publishes_only_current_serial_vs_spawn_certificate(tmp_path, monkeypatch):
    from caisen import daily_research
    from scripts import benchmark_current_rule_workers as runner

    symbols = [("stock", "%06d" % (600000 + index)) for index in range(36)]
    revision = "mtime_ns:123,size:456"
    inputs = [{"kind": kind, "code": code, "snapshot_revision": revision,
               "snapshot_path": "/frozen/" + code + ".csv",
               "registry_path": "/frozen/instruments.json", "registry_revision": "mtime_ns:12,size:34",
               "active_quote_manifest_path": "/frozen/active-quotes.json",
               "active_quote_manifest_revision": "mtime_ns:56,size:78",
               "tick_size": 0.01, "tick_size_source": "exchange_asset_class",
               "tick_rule_version": daily_research.TICK_RULE_VERSION}
              for kind, code in symbols]
    reference_manifest = {"status": "validated", "date_to": "2026-10-30", "protocol": "test-v1"}
    snapshot_manifest = {"snapshot_id": "test-snapshot", "paired_source_map_path": "test-map"}
    monkeypatch.setattr(runner, "_source_inputs",
                        lambda *_: (reference_manifest, snapshot_manifest, {}, inputs))

    clock = [0.0]
    run_results = []
    matrix_mode = [False]
    failed_worker_gate = {"worker": None, "gate": None}
    monkeypatch.setattr(runner.time, "perf_counter", lambda: clock[0])

    def fake_run_daily_research(*, output_dir, workers, symbols, recognition_config=None, **kwargs):
        duration = (20.0 if workers == 1 else 5.0)
        if matrix_mode[0]:
            duration = {1: 20.0, 2: 12.0, 4: 11.0, 6: 9.8, 8: 9.3, 10: 9.0}[workers]
        clock[0] += duration
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for kind, code in symbols:
            account_path = output_dir / (kind + "-" + code + ".json")
            payload = {
                "kind": kind,
                "code": code,
                "provenance": {"tick_size": 0.01,
                               "tick_size_source": "exchange_asset_class",
                               "tick_rule_version": daily_research.TICK_RULE_VERSION},
                "status": "complete",
                "signals": [{
                    "signal_date": "2026-01-02",
                    "entry_date": "2026-01-05",
                    "pattern_ids": [code + ":p1"],
                    "pattern_types": ["w_bottom"],
                    "pattern_tags": [{
                        "pattern_id": code + ":p1",
                        "instance_id": code + ":instance-v1",
                        "structure_id": code + ":structure-v1",
                        "boundary_version_id": "line-v1",
                        "frozen_boundary": {"slope": 0.01, "intercept": 10.0},
                        "events": [{"type": "confirmed", "occurred_at": "2026-01-02"}],
                    }],
                }],
                "pattern_events": [{"instance_id": code + ":instance-v1",
                                    "type": ("invalidated" if workers == failed_worker_gate["worker"]
                                             and failed_worker_gate["gate"] == "event" else "confirmed")}],
                "trades": [],
                "daily_equity": [],
                "rejections": [],
                "order_events": [],
                "ending_equity": 100000.0,
                "metrics": {},
                "accounting_checks": {
                    "cash_plus_position_equals_equity": not (
                        workers == failed_worker_gate["worker"]
                        and failed_worker_gate["gate"] == "accounting"),
                    "realized_pnl_matches_ending_equity": not (
                        workers == failed_worker_gate["worker"]
                        and failed_worker_gate["gate"] == "accounting"),
                },
            }
            account_path.write_text(json.dumps(payload), encoding="utf-8")
            rows.append({
                "kind": kind,
                "code": code,
                "status": "complete",
                "account_json": str(account_path),
                "provenance": {"data_revision": revision, "used_last_date": "2026-10-30"},
                "accounting_checks": payload["accounting_checks"],
            })
        result = {
            "status": "complete",
            "account_summaries": rows,
            "summary": {"account_count": len(rows)},
            "timings": {"total_seconds": duration},
            "parameters": {"recognition_config": list(recognition_config.cache_key())
                            if recognition_config is not None else {}},
            "recognition_experiment": daily_research._recognition_experiment_metadata(recognition_config),
        }
        manifest_path = output_dir / "run-manifest.json"
        manifest_path.write_text(json.dumps({
            "status": "complete",
            "parameters": result["parameters"],
            "recognition_experiment": result["recognition_experiment"],
        }), encoding="utf-8")
        result["artifacts"] = {"manifest_path": str(manifest_path)}
        result["test_recognition_config"] = recognition_config
        run_results.append(result)
        return result

    monkeypatch.setattr(daily_research, "run_daily_research", fake_run_daily_research)
    monkeypatch.setattr(daily_research, "_run_with_process_tree_peak",
                        lambda run: (run(), 1024))
    active_path = tmp_path / "benchmark-execution-v6.json"
    monkeypatch.setattr(daily_research, "BENCHMARK_PATH", active_path)
    monkeypatch.setattr(daily_research, "BENCHMARK_PROCESS_TREE_RSS_CAP", 4096)
    output_dir = tmp_path / "run"

    result = runner.run_policy(
        tmp_path / "reference.json", tmp_path / "snapshot", tmp_path / "universe.json",
        output_dir, candidate_workers=10, progress_callback=lambda _: None)

    assert result["status"] == "complete"
    assert result["policy_scope"] == "same-current-code-36-account-serial-vs-spawn"
    assert result["reference_count"] == 36
    assert result["recommended_workers"] == 10
    assert result["worker_counts"]["10"]["execution_mode"] == "spawn"
    assert result["event_parity_passed"] is True
    assert result["recognition_profile"] == {"name": "default", "cache_key": None,
                                             "recognition_experiment": None}
    assert result["recognition_profile_parity_passed"] is True
    assert run_results[0]["test_recognition_config"] is None
    assert run_results[1]["test_recognition_config"] is None
    assert result["source_unchanged_during_run"] is True
    assert result["data_revisions_match_references"] is True
    assert result["account_quote_provenance_matches_active_manifest"] is True
    assert result["speedup_vs_serial"] == 4.0
    assert active_path.exists()
    assert daily_research._validated_benchmark_config()["workers"] == 10
    reference = run_results[0]
    account_path = Path(reference["account_summaries"][0]["account_json"])
    account_payload = json.loads(account_path.read_text(encoding="utf-8"))
    account_payload["provenance"]["tick_size"] = 0.001
    account_path.write_text(json.dumps(account_payload), encoding="utf-8")
    assert runner._account_quotes_match(reference, inputs) is False

    matrix_mode[0] = True
    matrix_path = tmp_path / "matrix-certificate.json"
    monkeypatch.setattr(daily_research, "BENCHMARK_PATH", matrix_path)
    matrix_start = len(run_results)
    matrix = runner.run_policy(
        tmp_path / "reference.json", tmp_path / "snapshot", tmp_path / "universe.json",
        tmp_path / "matrix-run", candidate_workers=10, matrix=True,
        progress_callback=lambda _: None)
    assert len(run_results) - matrix_start == 6
    assert [result["timings"]["total_seconds"] for result in run_results[matrix_start:]] == [
        20.0, 12.0, 11.0, 9.8, 9.3, 9.0]
    assert set(matrix["worker_counts"]) == {"1", "2", "4", "6", "8", "10"}
    assert matrix["recommended_workers"] == 8
    assert matrix["eligible_workers_within_5_percent"] == [8, 10]
    assert matrix["candidate_ledger_signature_sha256"] == matrix["worker_counts"]["8"][
        "ledger_signature_sha256"]
    assert matrix["selected_process_tree_peak_bytes"] == matrix["worker_counts"]["8"][
        "process_tree_peak_bytes"]
    assert matrix_path.exists()

    for failed_gate in ("event", "accounting"):
        failed_worker_gate.update(worker=6, gate=failed_gate)
        failed_path = tmp_path / (failed_gate + "-failed-certificate.json")
        monkeypatch.setattr(daily_research, "BENCHMARK_PATH", failed_path)
        failed = runner.run_policy(
            tmp_path / "reference.json", tmp_path / "snapshot", tmp_path / "universe.json",
            tmp_path / (failed_gate + "-failed-run"), candidate_workers=10, matrix=True,
            progress_callback=lambda _: None)
        assert failed["status"] == "failed"
        assert failed["recommended_workers"] is None
        assert failed["worker_counts"]["6"]["candidate_gates_passed"] is False
        assert not failed_path.exists()

    failed_worker_gate.update(worker=None, gate=None)
    g1_certificate = tmp_path / "g1-independent-certificate.json"
    g1_output = tmp_path / "g1-run"
    run_start = len(run_results)
    g1_result = runner.run_policy(
        tmp_path / "reference.json", tmp_path / "snapshot", tmp_path / "universe.json",
        g1_output, candidate_workers=10, recognition_profile="g1",
        certificate_output=g1_certificate, progress_callback=lambda _: None)
    assert g1_result["status"] == "complete"
    assert g1_result["recognition_profile"]["name"] == "g1"
    assert g1_result["recognition_profile"]["cache_key"] == [
        "internal_rebound_coverage_g1_v1", None, "test_count_first", None, True, False]
    assert g1_result["recognition_profile_parity_passed"] is True
    assert g1_result["certificate_output"] == str(g1_certificate.resolve())
    assert g1_certificate.is_file()
    assert matrix_path.is_file()
    assert len(run_results) - run_start == 2
    assert all(run_results[index]["test_recognition_config"].require_internal_rebound_coverage is True
               for index in range(run_start, len(run_results)))

    serial_manifest = json.loads(Path(run_results[run_start]["artifacts"]["manifest_path"]).read_text())
    spawn_manifest = json.loads(Path(run_results[run_start + 1]["artifacts"]["manifest_path"]).read_text())
    assert serial_manifest["recognition_experiment"] == spawn_manifest["recognition_experiment"]
    assert serial_manifest["parameters"]["recognition_config"] == spawn_manifest["parameters"]["recognition_config"]

    real_run = daily_research.run_daily_research

    def fake_wrong_spawn_profile(**kwargs):
        result = real_run(**kwargs)
        if kwargs["workers"] > 1:
            manifest_path = Path(result["artifacts"]["manifest_path"])
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["parameters"]["recognition_config"] = ["wrong-profile"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return result

    monkeypatch.setattr(daily_research, "run_daily_research", fake_wrong_spawn_profile)
    wrong_profile = runner.run_policy(
        tmp_path / "reference.json", tmp_path / "snapshot", tmp_path / "universe.json",
        tmp_path / "wrong-profile-run", candidate_workers=10, recognition_profile="g1",
        certificate_output=tmp_path / "wrong-profile-certificate.json",
        progress_callback=lambda _: None)
    assert wrong_profile["status"] == "failed"
    assert wrong_profile["recognition_profile_parity_passed"] is False
    assert wrong_profile["worker_counts"]["10"]["candidate_gates_passed"] is False
    assert not (tmp_path / "wrong-profile-certificate.json").exists()


def test_policy_runner_rejects_serial_as_spawn_candidate(tmp_path):
    from scripts import benchmark_current_rule_workers as runner

    try:
        runner.run_policy("missing-manifest", "missing-snapshot", "missing-universe",
                          tmp_path / "run", candidate_workers=1)
    except ValueError as exc:
        assert "candidate_workers" in str(exc)
    else:
        raise AssertionError("serial must not be accepted as the parallel candidate")


def test_policy_runner_refuses_to_overwrite_existing_certificate(tmp_path, monkeypatch):
    from caisen import daily_research
    from scripts import benchmark_current_rule_workers as runner

    certificate = tmp_path / "existing-certificate.json"
    certificate.write_text('{"status":"old-measurement"}', encoding="utf-8")
    monkeypatch.setattr(daily_research, "BENCHMARK_PATH", certificate)
    output = tmp_path / "new-run"
    try:
        runner.run_policy("missing-reference", "missing-snapshot", "missing-universe",
                          output, candidate_workers=10)
    except FileExistsError as exc:
        assert "refusing to overwrite" in str(exc)
    else:
        raise AssertionError("an existing certificate must never be overwritten")
    assert not output.exists()
    assert json.loads(certificate.read_text(encoding="utf-8"))["status"] == "old-measurement"


def test_active_quote_migration_uses_current_loader_settings_without_csv_hashes(
        tmp_path, monkeypatch):
    from caisen.market import InstrumentFile
    from scripts import benchmark_current_rule_workers as runner

    snapshot_root = tmp_path / "snapshot"
    registry = snapshot_root / "caisen-system" / "config" / "instruments.json"
    registry.parent.mkdir(parents=True)
    registry.write_text("{}", encoding="utf-8")
    refs = []
    frozen_files = []
    instruments = []
    revisions = {}
    for index in range(36):
        kind = "stock" if index < 18 else "etf"
        code = "%06d" % ((600000 if kind == "stock" else 500000) + index)
        folder = snapshot_root / "data" / ("stocks" if kind == "stock" else "etfs")
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (code + ".csv")
        path.write_text("date,open,high,low,close,volume\n", encoding="utf-8")
        stat = path.stat()
        revision = "mtime_ns:%d,size:%d" % (stat.st_mtime_ns, stat.st_size)
        revisions[(kind, code)] = revision
        refs.append({"kind": kind, "code": code, "source": "archive", "account_path": "unused"})
        frozen_files.append({"kind": kind, "code": code, "snapshot_path": str(path),
                             "snapshot_data_revision": revision, "tick_size": 0.01,
                             "tick_size_source": "default"})
        instruments.append(InstrumentFile(code, kind, str(path)))
    reference = {"status": "validated", "date_to": "2026-10-30", "references": refs}
    snapshot = {"status": "complete", "account_count": 36, "date_to": "2026-10-30",
                "files": frozen_files}
    reference_path = tmp_path / "legacy-reference.json"
    snapshot_manifest_path = snapshot_root / "snapshot-manifest.json"
    reference_path.write_text(json.dumps(reference), encoding="utf-8")
    snapshot_manifest_path.write_text(json.dumps(snapshot), encoding="utf-8")
    legacy_reference_before = reference_path.read_bytes()
    previous_registry_env = "/pre-existing/instrument-registry.json"
    monkeypatch.setenv("CAISEN_INSTRUMENTS", previous_registry_env)
    monkeypatch.setattr(runner, "discover_instruments", lambda _: instruments)
    monkeypatch.setattr(runner, "_sha256", lambda _: (_ for _ in ()).throw(
        AssertionError("market CSV content hashes are not part of input validation")))

    def load_bars(instrument, root):
        return {"data_revision": revisions[(instrument.kind, instrument.code)],
                "tick_size": 0.001 if instrument.kind == "etf" else 0.01,
                "tick_size_source": "exchange_asset_class",
                "tick_rule_version": runner.research.TICK_RULE_VERSION,
                "normalized_rows": 0}

    monkeypatch.setattr(runner.research, "load_research_bars", load_bars)
    active_path = tmp_path / "active-quote-manifest.json"
    active = runner.prepare_active_quote_manifest(reference_path, snapshot_root, active_path)
    assert active["migration"]["kind"] == "active_loader_quote_rebase"
    assert active["migration"]["source_reference_manifest"] == str(reference_path.resolve())
    assert active["migration"]["migrated_quote_count"] == 36
    assert active["migration"]["tick_size_value_change_count"] == 18
    assert active["migration"]["tick_size_source_change_count"] == 36
    assert reference_path.read_bytes() == legacy_reference_before
    _, _, _, inputs = runner._source_inputs(active_path, snapshot_root)
    assert runner.os.environ.get("CAISEN_INSTRUMENTS") == previous_registry_env

    assert len(inputs) == 36
    assert all("snapshot_sha256" not in item for item in inputs)
    assert all(item["tick_size_source"] == "exchange_asset_class" for item in inputs)
    assert {item["tick_size"] for item in inputs if item["kind"] == "etf"} == {0.001}
    assert {item["tick_size"] for item in inputs if item["kind"] == "stock"} == {0.01}
    assert all(item["tick_rule_version"] == runner.research.TICK_RULE_VERSION for item in inputs)
    fingerprint = runner._input_fingerprint(inputs)
    runner._assert_inputs_unchanged(active_path, snapshot_root, fingerprint)
    changed_path = Path(inputs[0]["snapshot_path"])
    stat = changed_path.stat()
    import os
    os.utime(changed_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    try:
        runner._assert_inputs_unchanged(active_path, snapshot_root, fingerprint)
    except ValueError as exc:
        assert "snapshot revision mismatch" in str(exc)
    else:
        raise AssertionError("snapshot stat changes must fail revalidation")
    os.utime(changed_path, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    active["inputs"][0]["tick_size"] = 0.01
    active_path.write_text(json.dumps(active), encoding="utf-8")
    try:
        runner._source_inputs(active_path, snapshot_root)
    except ValueError as exc:
        assert "active quote manifest mismatch" in str(exc)
    else:
        raise AssertionError("active quote drift must fail preflight")
