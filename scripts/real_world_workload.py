#!/usr/bin/env python3
"""
Aurora DSQL real-world-style workload for indexed writes and lookup reads.

Experiments:
1. One multi-row INSERT with as many synthetic event rows as fit in ~1 MiB
   of logical row payload, measured after secondary indexes have been created.
2. Isolated single-query reads that mirror a production-style lookup shape.
"""

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from collect_cloudwatch_metrics import collect_metrics
from dsql_client import DSQLClient


TARGET_WRITE_BYTES = 1_048_576
COST_PER_DPU = 0.00000625

METRICS = [
    "TotalDPU",
    "WriteDPU",
    "ReadDPU",
    "ComputeDPU",
    "BytesWritten",
    "BytesRead",
    "ComputeTime",
    "TotalTransactions",
]

EVENT_COLUMNS = [
    "request_id",
    "event_token",
    "plan_name",
    "account_name",
    "host",
    "channel",
    "ref_domain",
    "account_id",
    "plan_id",
    "item_name",
    "item_width",
    "item_height",
    "browser_name",
    "browser_version",
    "browser_major",
    "device_model",
    "device_type",
    "device_vendor",
    "os_name",
    "os_version",
    "client_signature",
    "cpu_architecture",
    "network_address",
    "country_code",
    "region",
    "city",
    "location",
    "network_class",
    "actor_class",
    "path",
    "time",
    "redirect_url",
    "sub_account_id",
    "slot_id",
    "referrer",
    "client_hash",
    "data",
]

SAMPLES = [
    {
        "event_token": "Qm5V2gR1pALEJx4=",
        "network_address": "2.57.76.115",
        "client_signature": "Mozilla/5.0 (Linux; Android 15; SM-F711U Build/AP3A.240905.015.A2; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/140.0.7339.207 Mobile Safari/537.36",
        "plan_id": "5821",
        "slot_id": "374",
        "account_id": "7314",
        "plan_name": "5821_Sample Event Group_RSYNC_40629",
        "item_name": "Sample Event_40629",
        "browser_name": "Chrome WebView",
        "browser_version": "140.0.7339.207",
        "browser_major": "140",
        "device_model": "SM-F711U",
        "device_type": "mobile",
        "device_vendor": "Samsung",
        "os_name": "Android",
        "os_version": "15",
        "country_code": "QZ",
        "region": "QX",
        "city": "Redrock",
    },
    {
        "event_token": "Rk7M9p-YuBMEQs2=",
        "network_address": "2.57.78.23",
        "client_signature": "Mozilla/5.0 (X11; CrOS x86_64 14541.0.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
        "plan_id": "5822",
        "slot_id": "375",
        "account_id": "7314",
        "plan_name": "5822_Sample Event Group_RSYNC_42157",
        "item_name": "Sample Event_42157",
        "browser_name": "Chrome",
        "browser_version": "140.0.0.0",
        "browser_major": "140",
        "device_model": "Chromebook",
        "device_type": "desktop",
        "device_vendor": "Google",
        "os_name": "Chromium OS",
        "os_version": "14541",
        "country_code": "QZ",
        "region": "QX",
        "city": "Redrock",
    },
    {
        "event_token": "Tn3cBh6woDMEtL8=",
        "network_address": "2.57.79.44",
        "client_signature": "Mozilla/5.0 (iPhone; CPU iPhone OS 18_6_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148",
        "plan_id": "5823",
        "slot_id": "376",
        "account_id": "7314",
        "plan_name": "5823_Sample Event Group_RSYNC_38148",
        "item_name": "Sample Event_38148",
        "browser_name": "WebKit",
        "browser_version": "18.6.2",
        "browser_major": "18",
        "device_model": "iPhone",
        "device_type": "mobile",
        "device_vendor": "Apple",
        "os_name": "iOS",
        "os_version": "18.6.2",
        "country_code": "QZ",
        "region": "QX",
        "city": "Redrock",
    },
]

TARGET = {
    "request_id": "dsql-exp-target-request",
    "event_token": "dsql-exp-target-event-token",
    "account_id": "7314",
    "plan_id": "5821",
    "slot_id": "374",
    "network_address": "198.51.100.44",
    "client_signature": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
}


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def quote_ident(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def floor_minute(dt: datetime) -> datetime:
    return dt.replace(second=0, microsecond=0)


def ceil_minute(dt: datetime) -> datetime:
    floored = floor_minute(dt)
    if floored == dt:
        return floored
    return floored + timedelta(minutes=1)


def wait_for_fresh_metric_minute() -> datetime:
    now = datetime.now(timezone.utc)
    seconds = now.second + now.microsecond / 1_000_000
    sleep_seconds = 6 - seconds if seconds < 5 else 66 - seconds
    if sleep_seconds > 0:
        log(f"Waiting {sleep_seconds:.1f}s for a clean CloudWatch metric minute")
        time.sleep(sleep_seconds)
    return floor_minute(datetime.now(timezone.utc))


def metric_total(metrics: Dict[str, Dict[str, Any]], name: str) -> float:
    return float(metrics.get(name, {}).get("total", 0) or 0)


def collect_metrics_with_polling(
    cluster_id: str,
    region: str | None,
    start_time: datetime,
    end_time: datetime,
    *,
    max_wait_seconds: int = 900,
) -> Dict[str, Dict[str, Any]]:
    deadline = time.monotonic() + max_wait_seconds
    attempt = 1
    last_metrics: Dict[str, Dict[str, Any]] = {}

    while True:
        collected = collect_metrics(
            cluster_id=cluster_id,
            region=region,
            start_time=start_time,
            end_time=end_time,
            metrics=METRICS,
            period=60,
        )
        last_metrics = collected["metrics"]
        datapoints = sum(data.get("datapoints", 0) for data in last_metrics.values())
        total_dpu = metric_total(last_metrics, "TotalDPU")

        if datapoints > 0 and total_dpu > 0:
            log(f"CloudWatch metrics available after attempt {attempt}")
            return last_metrics

        if time.monotonic() >= deadline:
            log("CloudWatch metrics did not populate before timeout; recording latest response")
            return last_metrics

        log(f"Waiting 30s for CloudWatch metrics (attempt {attempt})")
        attempt += 1
        time.sleep(30)


def connect_with_retry(cluster_id: str, region: str | None = None):
    client = DSQLClient(log=log)
    for attempt in range(1, 5):
        try:
            conn = client.connect(cluster_id, region)
            execute_query(conn, "SELECT 1", fetch_rows=False)
            return conn
        except Exception as exc:
            message = str(exc)
            if "waking up cluster" in message and attempt < 4:
                log("DSQL cluster is waking up; retrying in 30s")
                time.sleep(30)
                continue
            raise


def execute_query(conn, query: str, params: Tuple[Any, ...] = (), *, fetch_rows: bool = True) -> list[tuple]:
    with conn.cursor() as cursor:
        cursor.execute(query, params)
        rows = cursor.fetchall() if fetch_rows and cursor.description else []
    conn.commit()
    return rows


def setup_schema(conn, schema: str) -> None:
    log(f"Creating schema {schema}")
    execute_query(conn, f"CREATE SCHEMA IF NOT EXISTS {quote_ident(schema)}", fetch_rows=False)
    execute_query(
        conn,
        f"""
        CREATE TABLE {quote_ident(schema)}.event_log (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            request_id TEXT,
            event_token TEXT,
            plan_name TEXT,
            account_name TEXT,
            host TEXT,
            channel TEXT,
            ref_domain TEXT,
            account_id TEXT NOT NULL,
            plan_id TEXT,
            item_name TEXT,
            item_width INTEGER,
            item_height INTEGER,
            browser_name TEXT,
            browser_version TEXT,
            browser_major TEXT,
            device_model TEXT,
            device_type TEXT,
            device_vendor TEXT,
            os_name TEXT,
            os_version TEXT,
            client_signature TEXT,
            cpu_architecture TEXT,
            network_address TEXT,
            country_code TEXT,
            region TEXT,
            city TEXT,
            location TEXT,
            network_class TEXT,
            actor_class TEXT,
            path TEXT,
            time TIMESTAMPTZ NOT NULL,
            redirect_url TEXT,
            sub_account_id TEXT,
            slot_id TEXT,
            referrer TEXT,
            client_hash TEXT,
            data TEXT
        )
        """,
        fetch_rows=False,
    )

    indexes = [
        (
            "event_lookup_token_time_idx",
            "(account_id, event_token, time)",
        ),
        (
            "event_lookup_network_client_time_idx",
            "(account_id, network_address, client_signature, time)",
        ),
        (
            "event_lookup_token_plan_slot_time_idx",
            "(account_id, event_token, plan_id, slot_id, time)",
        ),
        (
            "event_lookup_network_client_plan_slot_time_idx",
            "(account_id, network_address, client_signature, plan_id, slot_id, time)",
        ),
    ]

    for index_name, columns in indexes:
        log(f"Creating async index {index_name}")
        execute_query(
            conn,
            f"CREATE INDEX ASYNC {quote_ident(index_name)} "
            f"ON {quote_ident(schema)}.event_log {columns}",
            fetch_rows=False,
        )


def make_event_row(i: int, base_time: datetime) -> Dict[str, Any]:
    sample = SAMPLES[i % len(SAMPLES)]
    is_target = i == 0
    event_time = base_time - timedelta(minutes=i % (30 * 24 * 6))
    event_token = TARGET["event_token"] if is_target else f"dsql-exp-{i:08d}-{sample['event_token']}"
    request_id = TARGET["request_id"] if is_target else f"dsql-exp-request-{i:08d}"
    account_id = TARGET["account_id"] if is_target else sample["account_id"]
    plan_id = TARGET["plan_id"] if is_target else sample["plan_id"]
    slot_id = TARGET["slot_id"] if is_target else sample["slot_id"]
    network_address = TARGET["network_address"] if is_target else sample["network_address"]
    client_signature = TARGET["client_signature"] if is_target else sample["client_signature"]

    data = {
        "source": "aurora-dsql-experiment",
        "bucket_key": f"bucket-{i:08d}",
        "buffer_token": f"token-{i % 37:03d}",
        "site": "synthetic-source.test",
        "debug": {
            "macro_plan": plan_id,
            "macro_slot": slot_id,
            "slot": i % 12,
        },
    }

    return {
        "request_id": request_id,
        "event_token": event_token,
        "plan_name": sample["plan_name"],
        "account_name": "SampleOrg",
        "host": "events.pipeline.example.test",
        "channel": "network",
        "ref_domain": "example-source.test",
        "account_id": account_id,
        "plan_id": plan_id,
        "item_name": sample["item_name"],
        "item_width": 300,
        "item_height": 250,
        "browser_name": sample["browser_name"],
        "browser_version": sample["browser_version"],
        "browser_major": sample["browser_major"],
        "device_model": sample["device_model"],
        "device_type": sample["device_type"],
        "device_vendor": sample["device_vendor"],
        "os_name": sample["os_name"],
        "os_version": sample["os_version"],
        "client_signature": client_signature,
        "cpu_architecture": "arm64" if is_target else "x86_64",
        "network_address": network_address,
        "country_code": sample["country_code"],
        "region": sample["region"],
        "city": sample["city"],
        "location": "POINT(-90.0490 35.1495)",
        "network_class": "fixed-broadband",
        "actor_class": "standard-client",
        "path": "/event.gif",
        "time": event_time,
        "redirect_url": "https://example-destination.test/",
        "sub_account_id": "segment-a",
        "slot_id": slot_id,
        "referrer": "https://example-source.test/article",
        "client_hash": f"hash-{i % 1_000:04d}",
        "data": data,
    }


def row_logical_bytes(row: Dict[str, Any]) -> int:
    serializable = {
        key: value.isoformat() if isinstance(value, datetime) else value
        for key, value in row.items()
    }
    return len(json.dumps(serializable, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def build_rows_for_target(target_bytes: int) -> Tuple[List[Dict[str, Any]], int]:
    base_time = datetime.now(timezone.utc) - timedelta(minutes=5)
    rows: List[Dict[str, Any]] = []
    total_bytes = 0
    i = 0

    while True:
        row = make_event_row(i, base_time)
        size = row_logical_bytes(row)
        if rows and total_bytes + size > target_bytes:
            break
        rows.append(row)
        total_bytes += size
        i += 1

    return rows, total_bytes


def build_insert(rows: Sequence[Dict[str, Any]], schema: str) -> Tuple[str, Tuple[Any, ...], int]:
    placeholders = "(" + ", ".join(["%s"] * len(EVENT_COLUMNS)) + ")"
    values_sql = ", ".join([placeholders] * len(rows))
    column_sql = ", ".join(quote_ident(column) for column in EVENT_COLUMNS)
    query = f"INSERT INTO {quote_ident(schema)}.event_log ({column_sql}) VALUES {values_sql}"
    params: List[Any] = []

    for row in rows:
        for column in EVENT_COLUMNS:
            value = row[column]
            params.append(
                json.dumps(value, sort_keys=True, separators=(",", ":"))
                if column == "data"
                else value
            )

    return query, tuple(params), len(query.encode("utf-8"))


def build_lookup_query(
    schema: str,
    *,
    account_id: str,
    event_token: Optional[str],
    network_address: Optional[str],
    client_signature: Optional[str],
    plan_ids: Optional[Sequence[str]] = None,
    slot_ids: Optional[Sequence[str]] = None,
    window_days: int = 7,
) -> Tuple[str, Tuple[Any, ...]]:
    cutoff_time = datetime.now(timezone.utc) - timedelta(days=window_days)
    branches: List[str] = []
    params: List[Any] = []

    def add_filters() -> str:
        filter_sql = ""
        if plan_ids:
            filter_sql += " AND plan_id = ANY(%s)"
            params.append(list(plan_ids))
        if slot_ids:
            filter_sql += " AND slot_id = ANY(%s)"
            params.append(list(slot_ids))
        return filter_sql

    if event_token:
        params.extend([account_id, event_token, cutoff_time])
        branches.append(
            f"""
            SELECT plan_id, slot_id, match_time
            FROM (
                SELECT plan_id, slot_id, time AS match_time
                FROM {quote_ident(schema)}.event_log
                WHERE account_id = %s
                  AND event_token = %s
                  AND time >= %s
                  {add_filters()}
                ORDER BY time DESC
                LIMIT 1
            ) AS token_match
            """
        )

    if network_address and client_signature:
        params.extend([account_id, network_address, client_signature, cutoff_time])
        branches.append(
            f"""
            SELECT plan_id, slot_id, match_time
            FROM (
                SELECT plan_id, slot_id, time AS match_time
                FROM {quote_ident(schema)}.event_log
                WHERE account_id = %s
                  AND network_address = %s
                  AND client_signature = %s
                  AND time >= %s
                  {add_filters()}
                ORDER BY time DESC
                LIMIT 1
            ) AS network_client_match
            """
        )

    if not branches:
        raise ValueError("At least one match branch is required")

    union_sql = "\nUNION ALL\n".join(branches)
    query = f"""
        SELECT plan_id, slot_id
        FROM (
            {union_sql}
        ) AS candidates
        ORDER BY match_time DESC
        LIMIT 1
    """
    return query, tuple(params)


def explain_query(conn, query: str, params: Tuple[Any, ...]) -> str:
    try:
        rows = execute_query(conn, f"EXPLAIN (ANALYZE, VERBOSE, SETTINGS) {query}", params)
    except Exception as exc:
        return f"EXPLAIN failed: {exc}"
    return "\n".join(row[0] for row in rows)


def summarize_metrics(metrics: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    total_dpu = metric_total(metrics, "TotalDPU")
    return {
        "TotalDPU": total_dpu,
        "WriteDPU": metric_total(metrics, "WriteDPU"),
        "ReadDPU": metric_total(metrics, "ReadDPU"),
        "ComputeDPU": metric_total(metrics, "ComputeDPU"),
        "BytesWritten": metric_total(metrics, "BytesWritten"),
        "BytesRead": metric_total(metrics, "BytesRead"),
        "ComputeTime": metric_total(metrics, "ComputeTime"),
        "TotalTransactions": metric_total(metrics, "TotalTransactions"),
        "cost_usd": total_dpu * COST_PER_DPU,
    }


def measure_operation(
    conn,
    cluster_id: str,
    region: str | None,
    *,
    name: str,
    operation_type: str,
    query: str,
    params: Tuple[Any, ...],
    fetch_rows: bool,
) -> Dict[str, Any]:
    metric_start = wait_for_fresh_metric_minute()
    start_time = datetime.now(timezone.utc)
    log(f"Running {operation_type} experiment: {name}")
    result = execute_query(conn, query, params, fetch_rows=fetch_rows)
    end_time = datetime.now(timezone.utc)
    metric_end = ceil_minute(end_time) + timedelta(minutes=1)
    metrics = collect_metrics_with_polling(cluster_id, region, metric_start, metric_end)
    rows = result if fetch_rows and result else []

    return {
        "name": name,
        "operation_type": operation_type,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "metric_window_start": metric_start.isoformat(),
        "metric_window_end": metric_end.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
        "rows_returned": len(rows),
        "result_rows": [list(row) for row in rows],
        "metrics": summarize_metrics(metrics),
        "raw_metrics": metrics,
    }


def write_markdown_report(results: Dict[str, Any], path: Path) -> None:
    write = results["write_experiment"]
    reads = results["read_experiments"]
    lines = [
        "# Aurora DSQL Real-World-Style Workload",
        "",
        f"Run: {results['run_id']}",
        f"Cluster: `{results['cluster_id']}`",
        f"Schema: `{results['schema']}`",
        "",
        "## 1 MiB Multi-Row Event Insert",
        "",
        f"- Rows inserted: {write['row_count']:,}",
        f"- Logical row payload: {write['logical_payload_bytes']:,} bytes",
        f"- INSERT SQL text: {write['insert_sql_bytes']:,} bytes",
        "- Secondary indexes were created before this insert was measured.",
        f"- Duration: {write['duration_seconds']:.3f}s",
        f"- TotalDPU: {write['metrics']['TotalDPU']:.6f}",
        f"- WriteDPU: {write['metrics']['WriteDPU']:.6f}",
        f"- ReadDPU: {write['metrics']['ReadDPU']:.6f}",
        f"- ComputeDPU: {write['metrics']['ComputeDPU']:.6f}",
        f"- BytesWritten metric: {write['metrics']['BytesWritten']:.0f}",
        f"- Cost: ${write['metrics']['cost_usd']:.8f}",
        "",
        "## Single Read Scenarios",
        "",
        "| Scenario | Rows | Duration ms | TotalDPU | ReadDPU | ComputeDPU | BytesRead | Cost USD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for read in reads:
        metrics = read["metrics"]
        lines.append(
            "| "
            f"{read['name']} | "
            f"{read['rows_returned']} | "
            f"{read['duration_seconds'] * 1000:.2f} | "
            f"{metrics['TotalDPU']:.6f} | "
            f"{metrics['ReadDPU']:.6f} | "
            f"{metrics['ComputeDPU']:.6f} | "
            f"{metrics['BytesRead']:.0f} | "
            f"${metrics['cost_usd']:.8f} |"
        )

    lines.extend(
        [
            "",
            "## Indexes Tested",
            "",
            "```sql",
            f"CREATE INDEX ASYNC event_lookup_token_time_idx ON {results['schema']}.event_log (account_id, event_token, time);",
            f"CREATE INDEX ASYNC event_lookup_network_client_time_idx ON {results['schema']}.event_log (account_id, network_address, client_signature, time);",
            f"CREATE INDEX ASYNC event_lookup_token_plan_slot_time_idx ON {results['schema']}.event_log (account_id, event_token, plan_id, slot_id, time);",
            f"CREATE INDEX ASYNC event_lookup_network_client_plan_slot_time_idx ON {results['schema']}.event_log (account_id, network_address, client_signature, plan_id, slot_id, time);",
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    schema = args.schema or f"real_world_workload_run_{run_id}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = connect_with_retry(args.cluster_id)

    setup_schema(conn, schema)
    log(f"Waiting {args.index_wait_seconds}s for async indexes")
    time.sleep(args.index_wait_seconds)

    rows, logical_payload_bytes = build_rows_for_target(args.target_bytes)
    insert_query, insert_params, insert_sql_bytes = build_insert(rows, schema)
    log(
        "Prepared single INSERT with "
        f"{len(rows):,} rows and {logical_payload_bytes:,} logical payload bytes"
    )

    write_measurement = measure_operation(
        conn,
        args.cluster_id,
        None,
        name="one_insert_approx_1mib_events_with_indexes",
        operation_type="write",
        query=insert_query,
        params=insert_params,
        fetch_rows=False,
    )
    write_measurement["row_count"] = len(rows)
    write_measurement["logical_payload_bytes"] = logical_payload_bytes
    write_measurement["insert_sql_bytes"] = insert_sql_bytes
    write_measurement["target_bytes"] = args.target_bytes

    read_scenarios = [
        {
            "name": "token_and_network_hit_with_plan_slot",
            "event_token": TARGET["event_token"],
            "network_address": TARGET["network_address"],
            "client_signature": TARGET["client_signature"],
            "plan_ids": [TARGET["plan_id"]],
            "slot_ids": [TARGET["slot_id"]],
        },
        {
            "name": "token_hit_network_miss_with_plan_slot",
            "event_token": TARGET["event_token"],
            "network_address": "203.0.113.250",
            "client_signature": "NoMatch/1.0",
            "plan_ids": [TARGET["plan_id"]],
            "slot_ids": [TARGET["slot_id"]],
        },
        {
            "name": "network_client_hit_no_token_with_plan_slot",
            "event_token": None,
            "network_address": TARGET["network_address"],
            "client_signature": TARGET["client_signature"],
            "plan_ids": [TARGET["plan_id"]],
            "slot_ids": [TARGET["slot_id"]],
        },
        {
            "name": "token_and_network_hit_no_filters",
            "event_token": TARGET["event_token"],
            "network_address": TARGET["network_address"],
            "client_signature": TARGET["client_signature"],
            "plan_ids": None,
            "slot_ids": None,
        },
        {
            "name": "network_client_hit_no_token_no_filters",
            "event_token": None,
            "network_address": TARGET["network_address"],
            "client_signature": TARGET["client_signature"],
            "plan_ids": None,
            "slot_ids": None,
        },
        {
            "name": "both_branches_no_match",
            "event_token": "dsql-exp-no-such-event-token",
            "network_address": "203.0.113.251",
            "client_signature": "NoMatch/2.0",
            "plan_ids": [TARGET["plan_id"]],
            "slot_ids": [TARGET["slot_id"]],
        },
        {
            "name": "both_branches_no_match_no_filters",
            "event_token": "dsql-exp-no-such-event-token",
            "network_address": "203.0.113.251",
            "client_signature": "NoMatch/2.0",
            "plan_ids": None,
            "slot_ids": None,
        },
        {
            "name": "identity_hit_filter_no_match",
            "event_token": TARGET["event_token"],
            "network_address": TARGET["network_address"],
            "client_signature": TARGET["client_signature"],
            "plan_ids": ["plan-no-match"],
            "slot_ids": ["slot-no-match"],
        },
    ]

    read_results = []
    for scenario in read_scenarios:
        query, params = build_lookup_query(
            schema,
            account_id=TARGET["account_id"],
            event_token=scenario["event_token"],
            network_address=scenario["network_address"],
            client_signature=scenario["client_signature"],
            plan_ids=scenario["plan_ids"],
            slot_ids=scenario["slot_ids"],
            window_days=7,
        )
        measurement = measure_operation(
            conn,
            args.cluster_id,
            None,
            name=scenario["name"],
            operation_type="read",
            query=query,
            params=params,
            fetch_rows=True,
        )
        measurement["query_sql"] = query
        measurement["query_param_count"] = len(params)
        measurement["scenario"] = scenario
        log(f"Collecting EXPLAIN for {scenario['name']}")
        measurement["explain"] = explain_query(conn, query, params)
        read_results.append(measurement)

    results = {
        "run_id": run_id,
        "cluster_id": args.cluster_id,
        "region": None,
        "schema": schema,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "write_experiment": write_measurement,
        "read_experiments": read_results,
    }

    json_path = output_dir / f"real_world_workload_{run_id}.json"
    md_path = output_dir / f"real_world_workload_{run_id}.md"
    json_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    write_markdown_report(results, md_path)
    log(f"Wrote JSON results to {json_path}")
    log(f"Wrote Markdown report to {md_path}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster-id")
    parser.add_argument("--schema")
    parser.add_argument("--target-bytes", type=int, default=TARGET_WRITE_BYTES)
    parser.add_argument("--index-wait-seconds", type=int, default=180)
    parser.add_argument("--output-dir", default="results")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
