"""Aurora DSQL connection and query execution helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import boto3
import psycopg


@dataclass(frozen=True)
class DSQLConfig:
    database: str = "postgres"
    user: str = "admin"
    port: int = 5432
    connect_attempts: int = 6
    connect_timeout: int = 20
    connect_retry_seconds: int = 15


class DSQLClient:
    def __init__(self, config: DSQLConfig | None = None, log: Callable[[str], None] | None = None) -> None:
        self.config = config or DSQLConfig()
        self.session = boto3.Session()
        self.log = log or (lambda _message: None)

    def endpoint(self, cluster_id: str, region: str) -> str:
        return f"{cluster_id}.dsql.{region}.on.aws"

    def auth_token(self, cluster_id: str, region: str | None) -> tuple[str, str]:
        client = self.session.client("dsql", region_name=region)
        effective_region = client.meta.region_name
        endpoint = self.endpoint(cluster_id, effective_region)
        return client.generate_db_connect_admin_auth_token(endpoint, effective_region), endpoint

    def connect(self, cluster_id: str, region: str | None = None):
        last_error = None
        for attempt in range(1, self.config.connect_attempts + 1):
            token, endpoint = self.auth_token(cluster_id, region)
            try:
                return psycopg.connect(
                    host=endpoint,
                    port=self.config.port,
                    dbname=self.config.database,
                    user=self.config.user,
                    password=token,
                    sslmode="require",
                    autocommit=False,
                    connect_timeout=self.config.connect_timeout,
                )
            except psycopg.OperationalError as exc:
                last_error = exc
                if attempt == self.config.connect_attempts:
                    break
                self.log(
                    f"{cluster_id[:8]} connect attempt {attempt} failed; "
                    f"retrying in {self.config.connect_retry_seconds}s: {exc}"
                )
                time.sleep(self.config.connect_retry_seconds)
        raise last_error or RuntimeError(f"Could not connect to {cluster_id}")


def execute_query(conn, sql: str, fetch_mode: str = "all", explain: bool = False) -> tuple[int | None, str | None]:
    with conn.cursor() as cursor:
        cursor.execute(f"EXPLAIN (ANALYZE, SETTINGS) {sql}" if explain else sql)
        rows_returned = None
        explain_output = None
        if cursor.description:
            if explain:
                explain_output = "\n".join(row[0] for row in cursor.fetchall())
            elif fetch_mode == "all":
                rows_returned = len(cursor.fetchall())
            elif fetch_mode == "one":
                rows_returned = 1 if cursor.fetchone() else 0
            elif fetch_mode == "none":
                rows_returned = None
            else:
                raise ValueError(f"Unknown fetch mode: {fetch_mode}")
        conn.commit()
        return rows_returned, explain_output


def execute_sql(
    client: DSQLClient,
    cluster_id: str,
    region: str | None,
    sql: str,
    fetch_mode: str = "all",
    explain: bool = False,
    conn=None,
) -> tuple[float, int | None, str | None]:
    close_conn = conn is None
    conn = conn or client.connect(cluster_id, region)
    start = time.monotonic()
    rows_returned, explain_output = execute_query(conn, sql, fetch_mode=fetch_mode, explain=explain)
    elapsed = time.monotonic() - start
    if close_conn:
        conn.close()
    return elapsed, rows_returned, explain_output
