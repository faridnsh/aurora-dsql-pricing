"""Analyze Aurora DSQL JSONL measurement files and fit simple linear models."""

from __future__ import annotations

import csv
import glob
import json
import math
from pathlib import Path
from statistics import mean
from typing import Iterable

BYTES_PER_MB = 1024 * 1024


def load_jsonl(paths: Iterable[str]) -> list[dict]:
    records: list[dict] = []
    for path_text in paths:
        for path in glob.glob(path_text):
            with open(path, "r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
                    record["_source"] = path
                    records.append(normalize(record))
    return records


def metric_value(record: dict, name: str) -> float:
    if name in record and record[name] is not None:
        return float(record[name])

    nested = record.get("dpu_metrics") or record.get("metrics") or {}
    value = nested.get(name)
    if isinstance(value, dict):
        return float(value.get("total", 0.0) or 0.0)
    if value is not None:
        return float(value)

    cloudwatch = record.get("cloudwatch", {}).get("metrics", {})
    value = cloudwatch.get(name)
    if isinstance(value, dict):
        return float(value.get("total", 0.0) or 0.0)

    return 0.0


def normalize(record: dict) -> dict:
    normalized = dict(record)
    for name in [
        "TotalDPU",
        "ReadDPU",
        "WriteDPU",
        "ComputeDPU",
        "MultiRegionWriteDPU",
        "BytesRead",
        "BytesWritten",
        "ComputeTime",
    ]:
        normalized[name] = metric_value(record, name)

    if not normalized["TotalDPU"]:
        normalized["TotalDPU"] = (
            normalized["ReadDPU"]
            + normalized["WriteDPU"]
            + normalized["ComputeDPU"]
            + normalized["MultiRegionWriteDPU"]
        )

    normalized["size_bytes"] = float(normalized.get("size_bytes", 0.0) or 0.0)
    normalized["size_mb"] = normalized["size_bytes"] / BYTES_PER_MB
    normalized["bytes_read_mb"] = normalized["BytesRead"] / BYTES_PER_MB
    normalized["bytes_written_mb"] = normalized["BytesWritten"] / BYTES_PER_MB
    normalized["exec_time_seconds"] = float(normalized.get("exec_time_seconds", 0.0) or 0.0)
    return normalized


def finite_pair(x: float, y: float) -> bool:
    return math.isfinite(x) and math.isfinite(y)


def linear_regression(points: list[tuple[float, float]]) -> dict | None:
    clean = [(x, y) for x, y in points if finite_pair(x, y)]
    if len(clean) < 2:
        return None

    xs = [x for x, _ in clean]
    ys = [y for _, y in clean]
    x_mean = mean(xs)
    y_mean = mean(ys)
    ss_xx = sum((x - x_mean) ** 2 for x in xs)
    if ss_xx == 0:
        return None

    slope = sum((x - x_mean) * (y - y_mean) for x, y in clean) / ss_xx
    intercept = y_mean - slope * x_mean
    predicted = [slope * x + intercept for x in xs]
    ss_res = sum((y - y_hat) ** 2 for y, y_hat in zip(ys, predicted))
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    r_squared = 1.0 if ss_tot == 0 else 1 - ss_res / ss_tot
    return {
        "points": len(clean),
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
    }


def fit_models(records: list[dict]) -> list[dict]:
    specs = [
        ("WriteDPU vs size_mb", "size_mb", "WriteDPU"),
        ("ReadDPU vs size_mb", "size_mb", "ReadDPU"),
        ("TotalDPU vs size_mb", "size_mb", "TotalDPU"),
        ("WriteDPU vs bytes_written_mb", "bytes_written_mb", "WriteDPU"),
        ("ReadDPU vs bytes_read_mb", "bytes_read_mb", "ReadDPU"),
        ("ComputeDPU vs exec_time_seconds", "exec_time_seconds", "ComputeDPU"),
    ]

    models = []
    for label, x_key, y_key in specs:
        points = [
            (float(record.get(x_key, 0.0) or 0.0), float(record.get(y_key, 0.0) or 0.0))
            for record in records
            if record.get(y_key, 0.0) is not None
        ]
        model = linear_regression(points)
        if model:
            model.update({"model": label, "x": x_key, "y": y_key})
            models.append(model)
    return models


def print_summary(records: list[dict], models: list[dict], price_per_million_dpu: float) -> None:
    total_dpu = sum(record["TotalDPU"] for record in records)
    total_cost = total_dpu * price_per_million_dpu / 1_000_000
    print(f"Records:   {len(records)}")
    print(f"TotalDPU:  {total_dpu:,.4f}")
    print(f"Cost:      ${total_cost:,.8f} at ${price_per_million_dpu:g}/1M DPU")
    print()
    print("| Model | Points | Slope | Intercept | R^2 |")
    print("|---|---:|---:|---:|---:|")
    for model in models:
        print(
            f"| {model['model']} | {model['points']} | "
            f"{model['slope']:.8f} | {model['intercept']:.8f} | {model['r_squared']:.6f} |"
        )


def write_csv(path: str, models: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "x", "y", "points", "slope", "intercept", "r_squared"])
        writer.writeheader()
        writer.writerows(models)


