#!/usr/bin/env python3
"""Plot Aurora DSQL write metrics from enriched JSONL result files."""

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
    ("No PK", "no_pk", "o", "#2f6f9f"),
    ("Serial PK", "serial_pk", "s", "#3f8f4f"),
    ("UUID PK", "uuid_pk", "^", "#b44747"),
]

CHARTS = [
    (
        "BytesWritten",
        "BytesWritten, MB",
        1024 * 1024,
        "size_bytes",
        "Insert size, bytes",
        True,
        "write_limits_byteswritten_vs_insert_size.png",
    ),
    (
        "BytesRead",
        "BytesRead, MB",
        1024 * 1024,
        "size_bytes",
        "Insert size, bytes",
        True,
        "write_limits_bytesread_vs_insert_size.png",
    ),
    (
        "ComputeTime",
        "ComputeTime, seconds",
        1000,
        "size_bytes",
        "Insert size, bytes",
        True,
        "write_limits_computetime_vs_insert_size.png",
    ),
    (
        "WriteDPU",
        "WriteDPU",
        1,
        "BytesWritten",
        "BytesWritten, MB",
        False,
        "write_limits_writedpu_vs_byteswritten.png",
    ),
    (
        "ReadDPU",
        "ReadDPU",
        1,
        "BytesRead",
        "BytesRead, MB",
        False,
        "write_limits_readdpu_vs_bytesread.png",
    ),
    (
        "ComputeDPU",
        "ComputeDPU",
        1,
        "ComputeTime",
        "ComputeTime, seconds",
        False,
        "write_limits_computedpu_vs_computetime.png",
    ),
]

COMBINED_BYTES_CHART = "write_limits_bytes_io_vs_insert_size.png"
COMBINED_BYTES_LOGY_CHART = "write_limits_bytes_io_vs_insert_size_logy.png"
LARGE_WRITE_MIN_BYTES = 100_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create write-limit relationship charts from enriched Aurora DSQL "
            "JSONL results. The input files must contain BytesWritten, "
            "BytesRead, ComputeTime, WriteDPU, ReadDPU, and ComputeDPU."
        )
    )
    parser.add_argument("--no-pk", type=Path, required=True)
    parser.add_argument("--serial-pk", type=Path, required=True)
    parser.add_argument("--uuid-pk", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("assets"))
    parser.add_argument(
        "--exclude-size-bytes",
        type=int,
        default=10 * 1024 * 1024,
        help="Exact insert size to exclude. Default excludes the 10MB failed/outlier row.",
    )
    return parser.parse_args()


def source_paths(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "no_pk": args.no_pk,
        "serial_pk": args.serial_pk,
        "uuid_pk": args.uuid_pk,
    }


def load_records(paths: dict[str, Path], exclude_size_bytes: int) -> tuple[list[dict], list[str]]:
    records: list[dict] = []
    skipped: list[str] = []
    required = [
        "size_bytes",
        "BytesWritten",
        "BytesRead",
        "ComputeTime",
    ]
    dpu_required = ["WriteDPU", "ReadDPU", "ComputeDPU"]

    for label, key, _, _ in SOURCES:
        path = paths[key]
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip().startswith("{"):
                skipped.append(f"{path}:{line_number}: not a JSON object")
                continue

            record = json.loads(line)
            if record.get("operation_type") != "write":
                continue
            if not record.get("success"):
                skipped.append(f"{path}:{line_number}: {record.get('error', 'not successful')}")
                continue
            if record.get("size_bytes") == exclude_size_bytes:
                skipped.append(f"{path}:{line_number}: excluded size {exclude_size_bytes}")
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


def point_ticks(records: list[dict], field: str, divisor: int = 1) -> list[float]:
    ticks = sorted({float(record[field]) / divisor for record in records})
    return [tick for tick in ticks if tick > 0]


def human_bytes(value: float) -> str:
    if value >= 1024 * 1024:
        amount = value / (1024 * 1024)
        return f"{amount:.3g}MB"
    if value >= 1024:
        amount = value / 1024
        return f"{amount:.3g}KB"
    return f"{value:.3g}B"


def set_point_x_ticks(ax: plt.Axes, ticks: list[float], labels: list[str]) -> None:
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.xaxis.set_minor_formatter(NullFormatter())


def set_byte_y_ticks(ax: plt.Axes, values: np.ndarray) -> None:
    ticks = sorted({float(value) for value in values if value > 0})
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: human_bytes(value)))
    ax.yaxis.set_minor_formatter(NullFormatter())


def set_byte_y_formatter(ax: plt.Axes) -> None:
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: human_bytes(value)))
    ax.yaxis.set_minor_formatter(NullFormatter())


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


def can_fit(x_values: np.ndarray, y_values: np.ndarray) -> bool:
    return len(x_values) >= 2 and len(y_values) >= 2


def x_divisor(field: str) -> int:
    if field in {"BytesWritten", "BytesRead"}:
        return 1024 * 1024
    if field == "ComputeTime":
        return 1000
    return 1


def setup_axis(ax: plt.Axes, title: str, xlabel: str, ylabel: str, logx: bool) -> None:
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", alpha=0.28, linewidth=0.8)
    if logx:
        ax.set_xscale("log")


def plot_chart(records: list[dict], output_dir: Path, chart: tuple) -> None:
    y_field, y_label, y_divisor, x_field, x_label, logx, filename = chart
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    for label, _, marker, color in SOURCES:
        rows = [record for record in records if record["pattern"] == label]
        x_values = values(rows, x_field) / x_divisor(x_field)
        y_values = values(rows, y_field) / y_divisor
        ax.scatter(x_values, y_values, label=label, s=52, marker=marker, color=color, alpha=0.88)

        if not can_fit(x_values, y_values):
            continue
        slope, intercept, r_squared = fit_line(x_values, y_values)
        x_line = np.linspace(x_values.min(), x_values.max(), 200)
        ax.plot(
            x_line,
            slope * x_line + intercept,
            color=color,
            alpha=0.65,
            linewidth=1.6,
            label=f"{label} fit: y={slope:.4f}x+{intercept:.3f}, R2={r_squared:.4f}",
        )

    if logx and x_field == "size_bytes":
        x_axis_label = "Insert size (log scale)"
    else:
        x_axis_label = x_label + (" (log scale)" if logx else "")

    setup_axis(ax, f"{y_label} vs {x_label}", x_axis_label, y_label, logx)
    if logx and x_field == "size_bytes":
        ticks = point_ticks(records, "size_bytes")
        set_point_x_ticks(ax, ticks, [human_bytes(tick) for tick in ticks])
    ax.legend(fontsize=8)
    fig.savefig(output_dir / filename, dpi=180)
    plt.close(fig)


def plot_combined_bytes_io(records: list[dict], output_dir: Path, log_y: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    metric_styles = {
        "BytesWritten": {"color": "#d95f02", "linestyle": "-", "fill": True},
        "BytesRead": {"color": "#1b9e77", "linestyle": "--", "fill": False},
    }

    for label, _, marker, _ in SOURCES:
        rows = [record for record in records if record["pattern"] == label]
        x_values = values(rows, "size_bytes")
        written_mb = values(rows, "BytesWritten") / (1024 * 1024)
        read_mb = values(rows, "BytesRead") / (1024 * 1024)
        if len(x_values) == 0:
            continue

        ax.scatter(
            x_values,
            written_mb,
            label=f"{label} BytesWritten",
            s=52,
            marker=marker,
            color=metric_styles["BytesWritten"]["color"],
            edgecolors=metric_styles["BytesWritten"]["color"],
            alpha=0.9,
        )
        ax.scatter(
            x_values,
            read_mb,
            label=f"{label} BytesRead",
            s=52,
            marker=marker,
            facecolors="none",
            edgecolors=metric_styles["BytesRead"]["color"],
            linewidths=1.7,
            alpha=0.9,
        )

        if not can_fit(x_values, written_mb) or not can_fit(x_values, read_mb):
            continue
        written_slope, written_intercept, _ = fit_line(x_values, written_mb)
        read_slope, read_intercept, _ = fit_line(x_values, read_mb)
        x_line = np.linspace(x_values.min(), x_values.max(), 200)
        ax.plot(
            x_line,
            written_slope * x_line + written_intercept,
            color=metric_styles["BytesWritten"]["color"],
            linewidth=1.7,
            alpha=0.65,
        )
        ax.plot(
            x_line,
            read_slope * x_line + read_intercept,
            color=metric_styles["BytesRead"]["color"],
            linewidth=1.7,
            linestyle=metric_styles["BytesRead"]["linestyle"],
            alpha=0.65,
        )

    setup_axis(
        ax,
        "BytesWritten and BytesRead vs insert size" + (" (log y)" if log_y else ""),
        "Insert size (log scale)",
        "CloudWatch data (log scale)" if log_y else "CloudWatch bytes, MB",
        True,
    )
    if log_y:
        ax.set_yscale("log")
        set_mb_y_formatter(ax)
    ticks = point_ticks(records, "size_bytes")
    set_point_x_ticks(ax, ticks, [human_bytes(tick) for tick in ticks])
    metric_handles = [
        plt.Line2D(
            [0],
            [0],
            color=metric_styles["BytesWritten"]["color"],
            marker="o",
            linestyle="-",
            label="BytesWritten",
            markersize=7,
        ),
        plt.Line2D(
            [0],
            [0],
            color=metric_styles["BytesRead"]["color"],
            marker="o",
            markerfacecolor="none",
            linestyle="--",
            label="BytesRead",
            markersize=7,
        ),
    ]
    type_handles = [
        plt.Line2D(
            [0],
            [0],
            color="#444444",
            marker=marker,
            linestyle="",
            label=label,
            markersize=7,
        )
        for label, _, marker, _ in SOURCES
    ]
    metric_legend = ax.legend(handles=metric_handles, title="Metric", fontsize=8, loc="upper left")
    ax.add_artist(metric_legend)
    ax.legend(handles=type_handles, title="Write pattern", fontsize=8, loc="upper left", bbox_to_anchor=(0, 0.82))
    filename = COMBINED_BYTES_LOGY_CHART if log_y else COMBINED_BYTES_CHART
    fig.savefig(output_dir / filename, dpi=180)
    plt.close(fig)


def large_write_records(records: list[dict]) -> list[dict]:
    return [record for record in records if int(record["size_bytes"]) >= LARGE_WRITE_MIN_BYTES]


def plot_large_writes_byteswritten(records: list[dict], output_dir: Path, loglog: bool = False) -> None:
    rows_for_chart = large_write_records(records)
    if not rows_for_chart:
        return

    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)

    for label, _, marker, color in SOURCES:
        rows = [record for record in rows_for_chart if record["pattern"] == label]
        x_values = values(rows, "size_bytes")
        y_values = values(rows, "BytesWritten")
        ax.scatter(x_values, y_values, label=label, s=58, marker=marker, color=color, alpha=0.88)

        if not can_fit(x_values, y_values):
            continue
        slope, intercept, r_squared = fit_line(x_values, y_values)
        if loglog:
            x_line = np.geomspace(x_values.min(), x_values.max(), 200)
        else:
            x_line = np.linspace(x_values.min(), x_values.max(), 200)
        ax.plot(
            x_line,
            slope * x_line + intercept,
            color=color,
            alpha=0.65,
            linewidth=1.7,
            label=f"{label} fit: y={slope:.4f}x+{intercept:.0f}, R2={r_squared:.4f}",
        )

    title_suffix = " (log-log)" if loglog else ""
    setup_axis(
        ax,
        f"Large writes: BytesWritten vs insert size{title_suffix}",
        "Insert size" + (" (log scale)" if loglog else ""),
        "BytesWritten" + (" (log scale)" if loglog else ""),
        loglog,
    )
    ticks = point_ticks(rows_for_chart, "size_bytes")
    set_point_x_ticks(ax, ticks, [human_bytes(tick) for tick in ticks])
    if loglog:
        ax.set_yscale("log")
        set_byte_y_ticks(ax, values(rows_for_chart, "BytesWritten"))
    else:
        set_byte_y_formatter(ax)
    ax.legend(fontsize=8)
    filename = "large_writes_byteswritten_loglog.png" if loglog else "large_writes_byteswritten_linear.png"
    fig.savefig(output_dir / filename, dpi=180)
    plt.close(fig)


def plot_large_writes_writedpu(records: list[dict], output_dir: Path) -> None:
    rows_for_chart = large_write_records(records)
    if not rows_for_chart:
        return

    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)

    for label, _, marker, color in SOURCES:
        rows = [record for record in rows_for_chart if record["pattern"] == label]
        x_values = values(rows, "size_bytes")
        y_values = values(rows, "WriteDPU")
        ax.scatter(x_values, y_values, label=label, s=58, marker=marker, color=color, alpha=0.88)

        if not can_fit(x_values, y_values):
            continue
        slope, intercept, r_squared = fit_line(x_values, y_values)
        x_line = np.linspace(x_values.min(), x_values.max(), 200)
        ax.plot(
            x_line,
            slope * x_line + intercept,
            color=color,
            alpha=0.65,
            linewidth=1.7,
            label=f"{label} fit: y={slope:.8f}x+{intercept:.3f}, R2={r_squared:.4f}",
        )

    setup_axis(ax, "Large writes: WriteDPU vs insert size", "Insert size", "WriteDPU", False)
    ticks = point_ticks(rows_for_chart, "size_bytes")
    set_point_x_ticks(ax, ticks, [human_bytes(tick) for tick in ticks])
    ax.legend(fontsize=8)
    fig.savefig(output_dir / "large_writes_writedpu_linear.png", dpi=180)
    plt.close(fig)


def plot_overview(records: list[dict], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    for ax, chart in zip(axes.ravel(), CHARTS):
        y_field, y_label, y_divisor, x_field, x_label, logx, _ = chart
        for label, _, marker, color in SOURCES:
            rows = [record for record in records if record["pattern"] == label]
            x_values = values(rows, x_field) / x_divisor(x_field)
            y_values = values(rows, y_field) / y_divisor
            ax.scatter(x_values, y_values, label=label, s=34, marker=marker, color=color, alpha=0.85)

            if not can_fit(x_values, y_values):
                continue
            slope, intercept, _ = fit_line(x_values, y_values)
            x_line = np.linspace(x_values.min(), x_values.max(), 150)
            ax.plot(x_line, slope * x_line + intercept, color=color, alpha=0.55, linewidth=1.2)

        setup_axis(ax, f"{y_label} vs {x_label}", x_label + (" (log)" if logx else ""), y_label, logx)

    axes[0, 0].legend(title="Pattern", fontsize=8)
    fig.suptitle("Aurora DSQL write-limit metrics", fontsize=16, fontweight="bold")
    fig.savefig(output_dir / "write_limits_metric_relationships_overview.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with plt.xkcd(scale=0.9, length=90, randomness=2):
        plt.rcParams["font.family"] = ["Comic Sans MS", "DejaVu Sans"]
        records, skipped = load_records(source_paths(args), args.exclude_size_bytes)
        if not records:
            raise SystemExit("No usable records found")

        for chart in CHARTS:
            plot_chart(records, args.output_dir, chart)
        plot_combined_bytes_io(records, args.output_dir)
        plot_combined_bytes_io(records, args.output_dir, log_y=True)
        plot_large_writes_byteswritten(records, args.output_dir)
        plot_large_writes_byteswritten(records, args.output_dir, loglog=True)
        plot_large_writes_writedpu(records, args.output_dir)
        plot_overview(records, args.output_dir)

    print(f"Loaded {len(records)} records")
    for item in skipped:
        print(f"Skipped: {item}")
    print(f"Wrote charts to {args.output_dir}")


if __name__ == "__main__":
    main()
