#!/usr/bin/env python3
"""Plot Aurora DSQL read query metrics from JSONL result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402


SOURCES = [
    ("No PK single-row", "read_no_pk", "o", "#4c78a8"),
    ("Serial PK single-row", "read_serial_pk", "s", "#59a14f"),
    ("UUID PK single-row", "read_uuid_pk", "^", "#b07aa1"),
    ("SELECT * WHERE", "read_raw_where", "D", "#f28e2b"),
    ("SELECT * full scan", "read_raw_fullscan", "P", "#e15759"),
]

SERIAL_PK_BREAKDOWN_EXCLUDE_SIZE_BYTES = {10 * 1024 * 1024}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create read-query charts from Aurora DSQL JSONL results. "
            "The input records must contain size_bytes, BytesRead, ReadDPU, and ComputeTime."
        )
    )
    parser.add_argument("--read-no-pk", type=Path, required=True)
    parser.add_argument("--read-serial-pk", type=Path, required=True)
    parser.add_argument("--read-uuid-pk", type=Path, required=True)
    parser.add_argument("--read-raw-where", type=Path, required=True)
    parser.add_argument("--read-raw-fullscan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("assets"))
    return parser.parse_args()


def source_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "read_no_pk": args.read_no_pk,
        "read_serial_pk": args.read_serial_pk,
        "read_uuid_pk": args.read_uuid_pk,
        "read_raw_where": args.read_raw_where,
        "read_raw_fullscan": args.read_raw_fullscan,
    }


def load_records(paths: dict[str, Path]) -> tuple[list[dict], list[str]]:
    records: list[dict] = []
    skipped: list[str] = []
    required = ["size_bytes", "BytesRead", "ComputeTime"]
    dpu_required = ["ReadDPU"]

    for label, key, _, _ in SOURCES:
        path = paths[key]
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip().startswith("{"):
                skipped.append(f"{path}:{line_number}: not a JSON object")
                continue

            record = json.loads(line)
            if record.get("operation_type") != "read":
                skipped.append(f"{path}:{line_number}: operation_type={record.get('operation_type')}")
                continue
            if not record.get("success"):
                skipped.append(f"{path}:{line_number}: {record.get('error', 'not successful')}")
                continue

            missing = [field for field in required if field not in record]
            missing.extend(
                field
                for field in dpu_required
                if field not in record and f"CloudWatch{field}" not in record
            )
            if missing:
                skipped.append(f"{path}:{line_number}: missing {', '.join(missing)}")
                continue

            record = dict(record)
            record["pattern"] = label
            records.append(record)

    records.sort(key=lambda item: (item["pattern"], item["size_bytes"]))
    return records, skipped


def metric_value(record: dict, field: str) -> float:
    if field.endswith("DPU"):
        cloudwatch_field = f"CloudWatch{field}"
        if cloudwatch_field in record:
            return float(record[cloudwatch_field])
    return float(record[field])


def values(records: list[dict], field: str) -> np.ndarray:
    return np.array([metric_value(record, field) for record in records], dtype=float)


def human_bytes(value: float) -> str:
    if value >= 1024 * 1024:
        amount = value / (1024 * 1024)
        return f"{amount:g}MB"
    if value >= 1024:
        amount = value / 1024
        return f"{amount:g}KB"
    return f"{value:g}B"


def set_point_x_ticks(ax: plt.Axes, ticks: list[float], labels: list[str]) -> None:
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.xaxis.set_minor_formatter(NullFormatter())


def set_mb_y_formatter(ax: plt.Axes) -> None:
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: human_bytes(value * 1024 * 1024)))
    ax.yaxis.set_minor_formatter(NullFormatter())


def fit_line(x_values: np.ndarray, y_values: np.ndarray) -> tuple[float, float, float]:
    slope, intercept = np.polyfit(x_values, y_values, 1)
    predicted = slope * x_values + intercept
    residual_sum = float(np.sum((y_values - predicted) ** 2))
    total_sum = float(np.sum((y_values - np.mean(y_values)) ** 2))
    r_squared = 1.0 - residual_sum / total_sum if total_sum else 1.0
    return slope, intercept, r_squared


def setup_axis(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", alpha=0.28, linewidth=0.8)


def plot_output_size_vs_bytes_read(records: list[dict], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)

    for label, _, marker, color in SOURCES:
        rows = [record for record in records if record["pattern"] == label]
        output_mb = values(rows, "size_bytes") / (1024 * 1024)
        bytes_read_mb = values(rows, "BytesRead") / (1024 * 1024)
        ax.scatter(output_mb, bytes_read_mb, label=label, s=58, marker=marker, color=color, alpha=0.88)

    setup_axis(ax, "Query output size vs BytesRead", "Query output size (log scale)", "BytesRead (log scale)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    set_mb_y_formatter(ax)
    sizes = sorted({float(record["size_bytes"]) for record in records})
    ticks = [size / (1024 * 1024) for size in sizes if size > 0]
    set_point_x_ticks(ax, ticks, [human_bytes(size) for size in sizes if size > 0])
    ax.legend(fontsize=8)
    fig.savefig(output_dir / "read_output_size_vs_bytesread.png", dpi=180)
    plt.close(fig)


def plot_bytes_read_vs_read_dpu(records: list[dict], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)

    for label, _, marker, color in SOURCES:
        rows = [record for record in records if record["pattern"] == label]
        bytes_read_mb = values(rows, "BytesRead") / (1024 * 1024)
        read_dpu = values(rows, "ReadDPU")
        ax.scatter(bytes_read_mb, read_dpu, label=label, s=58, marker=marker, color=color, alpha=0.88)

    bytes_read_mb_all = values(records, "BytesRead") / (1024 * 1024)
    read_dpu_all = values(records, "ReadDPU")
    slope, intercept, r_squared = fit_line(bytes_read_mb_all, read_dpu_all)
    x_line = np.linspace(bytes_read_mb_all.min(), bytes_read_mb_all.max(), 300)
    ax.plot(
        x_line,
        slope * x_line + intercept,
        color="#333333",
        linewidth=1.8,
        alpha=0.7,
        label=f"all fit: ReadDPU={slope:.3f}*BytesRead_MB+{intercept:.3f}, R2={r_squared:.5f}",
    )

    setup_axis(ax, "BytesRead vs ReadDPU", "BytesRead, MB", "ReadDPU")
    ax.legend(fontsize=8)
    fig.savefig(output_dir / "read_bytesread_vs_readdpu.png", dpi=180)
    plt.close(fig)


def plot_compute_time_vs_bytes_read(records: list[dict], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)

    for label, _, marker, color in SOURCES:
        rows = [record for record in records if record["pattern"] == label]
        bytes_read_mb = values(rows, "BytesRead") / (1024 * 1024)
        compute_time_seconds = values(rows, "ComputeTime") / 1000
        ax.scatter(bytes_read_mb, compute_time_seconds, label=label, s=58, marker=marker, color=color, alpha=0.88)

    bytes_read_mb_all = values(records, "BytesRead") / (1024 * 1024)
    compute_time_seconds_all = values(records, "ComputeTime") / 1000
    slope, intercept, r_squared = fit_line(bytes_read_mb_all, compute_time_seconds_all)
    x_line = np.linspace(bytes_read_mb_all.min(), bytes_read_mb_all.max(), 300)
    ax.plot(
        x_line,
        slope * x_line + intercept,
        color="#333333",
        linewidth=1.8,
        alpha=0.7,
        label=f"all fit: ComputeTime_s={slope:.4f}*BytesRead_MB+{intercept:.3f}, R2={r_squared:.5f}",
    )

    setup_axis(ax, "ComputeTime vs BytesRead", "BytesRead, MB", "ComputeTime, seconds")
    ax.legend(fontsize=8)
    fig.savefig(output_dir / "read_computetime_vs_bytesread.png", dpi=180)
    plt.close(fig)


def plot_serial_pk_dpu_breakdown(records: list[dict], output_dir: Path) -> None:
    rows = [
        record
        for record in records
        if record["pattern"] == "Serial PK single-row"
        and str(record.get("test_id", "")).startswith("iter01_read_single_")
        and int(record["size_bytes"]) not in SERIAL_PK_BREAKDOWN_EXCLUDE_SIZE_BYTES
    ]
    if not rows:
        return

    rows.sort(key=lambda record: int(record["size_bytes"]))
    sizes_mb = values(rows, "size_bytes") / (1024 * 1024)
    read_dpu = values(rows, "ReadDPU")
    write_dpu = np.array(
        [
            metric_value(record, "WriteDPU")
            if "WriteDPU" in record or "CloudWatchWriteDPU" in record
            else 0
            for record in rows
        ],
        dtype=float,
    )
    compute_dpu = values(rows, "ComputeDPU")

    fig, ax = plt.subplots(figsize=(16, 7), constrained_layout=True)
    ax.fill_between(sizes_mb, 0, read_dpu, alpha=0.7, label="ReadDPU", color="#1f77b4")
    ax.fill_between(sizes_mb, read_dpu, read_dpu + write_dpu, alpha=0.7, label="WriteDPU", color="#ff7f0e")
    ax.fill_between(
        sizes_mb,
        read_dpu + write_dpu,
        read_dpu + write_dpu + compute_dpu,
        alpha=0.7,
        label="ComputeDPU",
        color="#2ca02c",
    )

    setup_axis(ax, "DPU Breakdown: Serial PK single-row reads", "Data size", "Total DPU")
    ax.set_xscale("log")
    sizes = [float(record["size_bytes"]) for record in rows]
    ticks = [size / (1024 * 1024) for size in sizes if size > 0]
    set_point_x_ticks(ax, ticks, [human_bytes(size) for size in sizes if size > 0])
    ax.legend(fontsize=12)
    fig.savefig(output_dir / "dpu_breakdown_serial_pk.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with plt.xkcd(scale=0.9, length=90, randomness=2):
        plt.rcParams["font.family"] = ["Comic Sans MS", "DejaVu Sans"]
        records, skipped = load_records(source_paths(args))
        if not records:
            raise SystemExit("No usable records found")

        plot_output_size_vs_bytes_read(records, args.output_dir)
        plot_bytes_read_vs_read_dpu(records, args.output_dir)
        plot_compute_time_vs_bytes_read(records, args.output_dir)
        plot_serial_pk_dpu_breakdown(records, args.output_dir)

    print(f"Loaded {len(records)} records")
    for item in skipped:
        print(f"Skipped: {item}")
    print(f"Wrote charts to {args.output_dir}")


if __name__ == "__main__":
    main()
