#!/usr/bin/env python3
"""Rerun the Aurora DSQL DPU benchmarks used by BLOG_POST.md."""

from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from collect_cloudwatch_metrics import collect_metrics, summarize
from generate_workload_sql import BenchmarkCase, BenchmarkGroup, blog_groups
from dsql_client import DSQLClient, DSQLConfig, execute_sql

METRIC_NAMES = ["WriteDPU", "ReadDPU", "ComputeDPU", "BytesWritten", "BytesRead", "ComputeTime"]
SETUP_GAP_SECONDS = 60
CLOUDWATCH_WAIT_SECONDS = 180
CLOUDWATCH_PERIOD_SECONDS = 60


@dataclass(frozen=True)
class Cluster:
    number: int
    cluster_id: str
    region: str


def default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path("results") / f"blog_benchmarks_{stamp}"


class BlogBenchmarkRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.output_dir = args.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=False)
        (self.output_dir / "logs").mkdir()
        self.clusters = self.load_clusters(args.clusters_file)
        self.client = DSQLClient(
            DSQLConfig(),
            log=lambda message: self.log("connection", message),
        )
        self.connections = {}
        self.connection_lock = threading.Lock()
        self.log_lock = threading.Lock()

    def load_clusters(self, path: Path) -> list[Cluster]:
        data = json.loads(path.read_text(encoding="utf-8"))
        clusters = [
            Cluster(
                number=int(item.get("global_number") or item.get("number") or index + 1),
                cluster_id=item["cluster_id"],
                region=item["region"],
            )
            for index, item in enumerate(data)
        ]
        if not clusters:
            raise SystemExit(f"No clusters found in {path}")
        return clusters

    def log(self, name: str, message: str) -> None:
        text = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        with self.log_lock:
            print(text, flush=True)
            with (self.output_dir / "logs" / f"{name}.log").open("a", encoding="utf-8") as handle:
                handle.write(text + "\n")

    def cluster_connection(self, cluster: Cluster):
        with self.connection_lock:
            conn = self.connections.get(cluster.cluster_id)
            if conn and not conn.closed:
                return conn
        conn = self.client.connect(cluster.cluster_id, cluster.region)
        with self.connection_lock:
            existing = self.connections.get(cluster.cluster_id)
            if existing and not existing.closed:
                conn.close()
                return existing
            self.connections[cluster.cluster_id] = conn
        return conn

    def execute_setup_sql(self, cluster: Cluster, statements: tuple[str, ...]) -> None:
        conn = self.cluster_connection(cluster)
        for statement in statements:
            execute_sql(self.client, cluster.cluster_id, cluster.region, statement, fetch_mode="none", conn=conn)

    def run_group(self, group: BenchmarkGroup) -> Path:
        group_clusters = self.clusters[group.cluster_start :] or self.clusters
        if len(group.cases) > len(group_clusters):
            self.log(group.name, f"WARNING: {len(group.cases)} cases but only {len(group_clusters)} clusters; metrics may aggregate.")
        assigned = [
            (index, case, group_clusters[index % len(group_clusters)])
            for index, case in enumerate(group.cases)
        ]
        used = {cluster.cluster_id: cluster for _, _, cluster in assigned}
        output = self.output_dir / f"{group.name}_results.jsonl"

        self.log(group.name, f"Starting {group.name}: {len(group.cases)} cases, output={output}")
        start_time = datetime.now(timezone.utc)

        setup_jobs = []
        if group.setup_sql:
            needed_clusters = group_clusters[: min(len(group.cases), len(group_clusters))]
            setup_jobs.extend((cluster, group.setup_sql) for cluster in needed_clusters)
        setup_jobs.extend((cluster, case.setup_sql) for _, case, cluster in assigned if case.setup_sql)

        if setup_jobs:
            self.log(group.name, f"Setting up {len(setup_jobs)} cluster/case targets in parallel")
            with ThreadPoolExecutor(max_workers=len(setup_jobs)) as executor:
                future_to_cluster = {
                    executor.submit(self.execute_setup_sql, cluster, statements): cluster
                    for cluster, statements in setup_jobs
                }
                for future in as_completed(future_to_cluster):
                    cluster = future_to_cluster[future]
                    future.result()
                    self.log(group.name, f"  setup complete on {cluster.number}:{cluster.cluster_id[:8]} ({cluster.region})")
            self.log(group.name, f"Waiting {SETUP_GAP_SECONDS}s after setup")
            time.sleep(SETUP_GAP_SECONDS)

        self.log(group.name, f"Running {len(group.cases)} cases in parallel")
        records: list[dict | None] = [None] * len(group.cases)
        with ThreadPoolExecutor(max_workers=len(group.cases)) as executor:
            future_to_index = {
                executor.submit(self.run_case, group.name, index, len(group.cases), case, cluster): index
                for index, case, cluster in assigned
            }
            for future in as_completed(future_to_index):
                records[future_to_index[future]] = future.result()

        self.log(group.name, f"Waiting {CLOUDWATCH_WAIT_SECONDS}s for CloudWatch")
        time.sleep(CLOUDWATCH_WAIT_SECONDS)
        end_time = datetime.now(timezone.utc)
        metrics = self.collect_metrics(group.name, used, start_time, end_time)

        with output.open("w", encoding="utf-8") as handle:
            for record in records:
                if record is None:
                    continue
                record.update(metrics.get(record["cluster_id"], {}))
                handle.write(json.dumps(record, sort_keys=True) + "\n")

        self.log(group.name, f"Wrote {output}")
        return output

    def run_case(self, name: str, index: int, total: int, case: BenchmarkCase, cluster: Cluster) -> dict:
        record = {
            "test_id": f"iter01_{case.test_id}",
            "cluster_id": cluster.cluster_id,
            "cluster_number": cluster.number,
            "region": cluster.region,
            "description": case.description,
            "size_bytes": case.size_bytes,
            "operation_type": case.operation_type,
            "query_length": len(case.sql),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "success": False,
        }
        self.log(name, f"Running {index + 1}/{total} on {cluster.number}:{cluster.cluster_id[:8]} - {case.description}")
        try:
            conn = self.cluster_connection(cluster)
            elapsed, rows_returned, explain_output = execute_sql(
                self.client,
                cluster.cluster_id,
                cluster.region,
                case.sql,
                fetch_mode="all" if case.fetch else "none",
                explain=case.explain,
                conn=conn,
            )
            record["exec_time_seconds"] = elapsed
            record["timestamp_end"] = datetime.now(timezone.utc).isoformat()
            record["success"] = True
            if rows_returned is not None:
                record["rows_returned"] = rows_returned
            if explain_output is not None:
                record["explain_output"] = explain_output
        except Exception as exc:
            record["timestamp_end"] = datetime.now(timezone.utc).isoformat()
            record["error"] = str(exc)
            self.log(name, f"  FAILED {cluster.number}:{cluster.cluster_id[:8]} {case.test_id}: {exc}")
        return record

    def collect_metrics(
        self,
        name: str,
        used: dict[str, Cluster],
        start_time: datetime,
        end_time: datetime,
    ) -> dict[str, dict[str, float]]:
        self.log(name, f"Collecting CloudWatch metrics for {len(used)} clusters in parallel")

        def collect(cluster: Cluster) -> tuple[str, dict[str, float]]:
            collected = collect_metrics(
                cluster_id=cluster.cluster_id,
                region=cluster.region,
                start_time=start_time,
                end_time=end_time,
                metrics=METRIC_NAMES,
                period=CLOUDWATCH_PERIOD_SECONDS,
            )
            return cluster.cluster_id, summarize(collected)

        results = {}
        clusters = list(used.values())
        with ThreadPoolExecutor(max_workers=len(clusters)) as executor:
            future_to_cluster = {executor.submit(collect, cluster): cluster for cluster in clusters}
            for future in as_completed(future_to_cluster):
                cluster_id, metrics = future.result()
                results[cluster_id] = metrics
        return results

    def close(self) -> None:
        for conn in list(self.connections.values()):
            conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clusters-file", type=Path, default=Path("../clusters_multiregion.json"))
    parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    parser.add_argument(
        "--only",
        choices=["all", "writes", "single-reads", "raw-where", "raw-fullscan", "compute"],
        default="all",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.clusters_file.exists():
        raise SystemExit(f"Cluster file not found: {args.clusters_file}")

    runner = BlogBenchmarkRunner(args)
    try:
        for group in blog_groups(args.only):
            runner.run_group(group)
    finally:
        runner.close()

    print(f"Benchmark results written to {runner.output_dir}")


if __name__ == "__main__":
    main()
