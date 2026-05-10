from __future__ import annotations

from common.io import read_json, write_csv, write_json


def test_write_and_read_json(tmp_path):
    path = tmp_path / "nested" / "payload.json"
    payload = {"name": "cnn", "score": 0.91}

    write_json(path, payload)

    assert read_json(path) == payload


def test_write_csv(tmp_path):
    path = tmp_path / "metrics.csv"

    write_csv(path, [{"model": "baseline", "macro_f1": 0.5}], ["model", "macro_f1"])

    assert path.read_text(encoding="utf-8").splitlines() == [
        "model,macro_f1",
        "baseline,0.5",
    ]
