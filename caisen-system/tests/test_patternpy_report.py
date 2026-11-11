import importlib.util
import json
import subprocess
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).parents[1] / "scripts" / "build_patternpy_600108_report.py"
SPEC = importlib.util.spec_from_file_location("patternpy_report", SCRIPT)
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def make_source_repo(tmp_path, monkeypatch):
    root = tmp_path / "PatternPy-fixture"
    source = root / "tradingpatterns"
    source.mkdir(parents=True)
    (source / "tradingpatterns.py").write_text(
        """import numpy as np
def detect_double_top_bottom(df, window=3, threshold=.05):
    df['double_pattern'] = np.nan
    if len(df) > 2: df.loc[2, 'double_pattern'] = 'Double Bottom'
    return df
def detect_head_shoulder(df, window=3):
    df['head_shoulder_pattern'] = np.nan
    if len(df) > 3: df.loc[3, 'head_shoulder_pattern'] = 'Inverse Head and Shoulder'
    return df
""",
        encoding="utf-8",
    )
    (root / "LICENSE.md").write_text("fixture license", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "fixture"], check=True)
    commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    monkeypatch.setattr(report, "PINNED_COMMIT", commit)
    return root


def make_csv(tmp_path):
    csv = tmp_path / "600108.csv"
    pd.DataFrame({
        "date": ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"],
        "code": ["600108"] * 5, "name": ["Test"] * 5,
        "open": [10, 9, 8, 9, 10], "high": [11, 10, 9, 10, 11],
        "low": [9, 8, 7, 8, 9], "close": [10, 9, 8, 9, 10],
        "volume": [100, 110, 120, 130, 140],
    }).to_csv(csv, index=False)
    return csv


def test_hits_are_raw_labels_with_next_bar_known_date_not_shapes(tmp_path, monkeypatch):
    source = make_source_repo(tmp_path, monkeypatch)
    bars, events, metadata = report.analyze(make_csv(tmp_path), source, windows=[3])
    dates = [bar["date"] for bar in bars]
    assert [(event["label"], event["date"], event["known_date"]) for event in events] == [
        ("Double Bottom", "2024-01-04", "2024-01-05"),
        ("Inverse Head and Shoulder", "2024-01-05", "2024-01-08"),
    ]
    for event in events:
        assert event["known_date"] == dates[dates.index(event["date"]) + 1]
        assert "anchor" not in event
        assert "neckline" not in event
    assert metadata["rows"] == 5


def test_html_is_offline_and_carries_events_and_provenance(tmp_path, monkeypatch):
    source = make_source_repo(tmp_path, monkeypatch)
    csv = make_csv(tmp_path)
    html_path, json_path = tmp_path / "report.html", tmp_path / "report.json"
    report.build_report(html_path, json_path, csv, source, [3])
    html = html_path.read_text()
    assert "cdn.jsdelivr.net" not in html
    assert "Double Bottom" in html
    assert "Inverse Head and Shoulder" in html
    assert report.PINNED_COMMIT in html
    assert json.loads(json_path.read_text())["events"]


