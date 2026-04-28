"""Generate the SQL workload used by the Aurora DSQL DPU blog benchmarks."""

from __future__ import annotations

import argparse
import random
import string
import uuid
from dataclasses import dataclass, field

WRITE_SIZES = [2**i for i in range(21)] + [2 * 1024 * 1024, 5 * 1024 * 1024]
RAW_ROW_BYTES = 1_048_473
RAW_ROW_COUNTS = [1, 2, 4, 8, 16, 32, 64, 128]
RAW_COLUMN_BYTES = 524_185


@dataclass(frozen=True)
class BenchmarkCase:
    test_id: str
    description: str
    operation_type: str
    size_bytes: int
    sql: str
    fetch: bool = False
    explain: bool = False
    setup_sql: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class BenchmarkGroup:
    name: str
    cases: tuple[BenchmarkCase, ...]
    setup_sql: tuple[str, ...] = field(default_factory=tuple)
    cluster_start: int = 0


def size_label(size: int) -> str:
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{size // 1024}KB"
    return f"{size // (1024 * 1024)}MB"


def payload(size: int) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(random.choice(alphabet) for _ in range(size))


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def schema_sql() -> tuple[str, ...]:
    return ("CREATE SCHEMA IF NOT EXISTS dpu_test",)


def write_setup_sql(pattern: str) -> tuple[str, ...]:
    if pattern == "no_pk":
        return schema_sql() + (
            "DROP TABLE IF EXISTS dpu_test.write_no_pk_test CASCADE",
            "CREATE TABLE dpu_test.write_no_pk_test (data TEXT)",
        )
    if pattern == "serial_pk":
        return schema_sql() + (
            "DROP TABLE IF EXISTS dpu_test.write_serial_pk_test CASCADE",
            "CREATE TABLE dpu_test.write_serial_pk_test (id INTEGER PRIMARY KEY, data TEXT)",
        )
    if pattern == "uuid_pk":
        return schema_sql() + (
            "DROP TABLE IF EXISTS dpu_test.write_uuid_pk_test CASCADE",
            "CREATE TABLE dpu_test.write_uuid_pk_test (id TEXT PRIMARY KEY, data TEXT)",
        )
    raise ValueError(f"Unknown write pattern: {pattern}")


def write_cases(pattern: str) -> tuple[BenchmarkCase, ...]:
    cases: list[BenchmarkCase] = []
    next_id = 1
    table = {"no_pk": "write_no_pk_test", "serial_pk": "write_serial_pk_test", "uuid_pk": "write_uuid_pk_test"}[pattern]
    columns = {"no_pk": "(data)", "serial_pk": "(id, data)", "uuid_pk": "(id, data)"}[pattern]

    for size in WRITE_SIZES:
        row_sizes = [size] if size <= 1024 * 1024 else [1024 * 1024] * (size // (1024 * 1024))
        values = []
        for row_size in row_sizes:
            if pattern == "no_pk":
                values.append(f"({sql_literal(payload(row_size))})")
            elif pattern == "serial_pk":
                data_size = max(1, row_size - 4)
                values.append(f"({next_id}, {sql_literal(payload(data_size))})")
                next_id += 1
            else:
                data_size = max(1, row_size - 36)
                values.append(f"({sql_literal(str(uuid.uuid4()))}, {sql_literal(payload(data_size))})")

        label = size_label(size)
        if size <= 1024 * 1024:
            test_id = f"write_single_{label}"
            description = {
                "no_pk": f"{label} single column write (no primary key)",
                "serial_pk": f"{label} row (data={max(1, size - 4)}B + SERIAL PK=4B)",
                "uuid_pk": f"{label} row (data={max(1, size - 36)}B + UUID PK=36B)",
            }[pattern]
        else:
            mb = size // (1024 * 1024)
            test_id = f"write_multi_{mb}rows_{mb}MB"
            description = f"{mb} rows x 1MB = {mb}MB total ({pattern})"

        cases.append(
            BenchmarkCase(
                test_id=test_id,
                description=description,
                operation_type="write",
                size_bytes=size,
                sql=f"INSERT INTO dpu_test.{table} {columns} VALUES {', '.join(values)}",
                explain=True,
            )
        )
    return tuple(cases)


def write_group(pattern: str) -> BenchmarkGroup:
    return BenchmarkGroup(
        name={"no_pk": "write_no_pk", "serial_pk": "write_serial_pk", "uuid_pk": "write_uuid_pk"}[pattern],
        setup_sql=write_setup_sql(pattern),
        cases=write_cases(pattern),
    )


def verify_table_sql(table: str) -> tuple[str, ...]:
    return (f"SELECT COUNT(*) FROM dpu_test.{table}",)


def single_read_cases(pattern: str) -> tuple[BenchmarkCase, ...]:
    table = {"no_pk": "write_no_pk_test", "serial_pk": "write_serial_pk_test", "uuid_pk": "write_uuid_pk_test"}[pattern]
    cases: list[BenchmarkCase] = []
    for index, size in enumerate([2**i for i in range(21)], start=1):
        label = size_label(size)
        if pattern == "serial_pk":
            sql = f"SELECT * FROM dpu_test.{table} WHERE id = {index}"
            description = f"Read {label} row by PK"
        else:
            sql = f"SELECT * FROM dpu_test.{table} LIMIT 1 OFFSET {index - 1}"
            description = f"Read {label} row"
        cases.append(BenchmarkCase(f"read_single_{label}", description, "read", size, sql, fetch=True, explain=True))

    if pattern == "serial_pk":
        extras = [
            ("read_multi_21rows", "Read all 21 single-row records by PK range", sum(2**i for i in range(21)), f"SELECT * FROM dpu_test.{table} WHERE id <= 21"),
            ("read_multi_2MB", "Read 2MB multi-row record (2 rows) by PK range", 2 * 1024 * 1024, f"SELECT * FROM dpu_test.{table} WHERE id BETWEEN 22 AND 23"),
            ("read_multi_5MB", "Read 5MB multi-row record (5 rows) by PK range", 5 * 1024 * 1024, f"SELECT * FROM dpu_test.{table} WHERE id BETWEEN 24 AND 28"),
        ]
    else:
        extras = [
            ("read_multi_21rows", "Read all 21 single-row records", sum(2**i for i in range(21)), f"SELECT * FROM dpu_test.{table} LIMIT 21"),
            ("read_multi_2MB", "Read 2MB multi-row record (2 rows)", 2 * 1024 * 1024, f"SELECT * FROM dpu_test.{table} LIMIT 2 OFFSET 21"),
            ("read_multi_5MB", "Read 5MB multi-row record (5 rows)", 5 * 1024 * 1024, f"SELECT * FROM dpu_test.{table} LIMIT 5 OFFSET 23"),
        ]
    cases.extend(BenchmarkCase(test_id, description, "read", size, sql, fetch=True, explain=True) for test_id, description, size, sql in extras)
    return tuple(cases)


def single_read_group(pattern: str) -> BenchmarkGroup:
    table = {"no_pk": "write_no_pk_test", "serial_pk": "write_serial_pk_test", "uuid_pk": "write_uuid_pk_test"}[pattern]
    return BenchmarkGroup(
        name={"no_pk": "read_no_pk", "serial_pk": "read_serial_pk", "uuid_pk": "read_uuid_pk"}[pattern],
        setup_sql=verify_table_sql(table),
        cases=single_read_cases(pattern),
    )


def raw_where_setup_sql() -> tuple[str, ...]:
    statements = [
        *schema_sql(),
        "DROP TABLE IF EXISTS dpu_test.read_raw_test",
        "CREATE TABLE dpu_test.read_raw_test (id INTEGER PRIMARY KEY, text_col1 TEXT, text_col2 TEXT)",
    ]
    for batch in range(13):
        start_id = batch * 10
        end_id = min(start_id + 9, 127)
        statements.append(
            f"""
            INSERT INTO dpu_test.read_raw_test (id, text_col1, text_col2)
            SELECT i, repeat('x', {RAW_COLUMN_BYTES}) || i::text, repeat('y', {RAW_COLUMN_BYTES}) || i::text
            FROM generate_series({start_id}, {end_id}) AS i
            """
        )
    return tuple(statements)


def raw_where_group() -> BenchmarkGroup:
    return BenchmarkGroup(
        name="read_raw_data",
        setup_sql=raw_where_setup_sql(),
        cluster_start=50,
        cases=tuple(
            BenchmarkCase(
                f"read_raw_{rows}rows",
                f"Read {rows} row{'s' if rows != 1 else ''} with SELECT * using WHERE",
                "read",
                rows * RAW_ROW_BYTES,
                f"SELECT * FROM dpu_test.read_raw_test WHERE id < {rows}",
                fetch=True,
                explain=True,
            )
            for rows in RAW_ROW_COUNTS
        ),
    )


def fullscan_setup_sql(rows: int) -> tuple[str, ...]:
    table = f"read_raw_fullscan_{rows}rows"
    statements = [
        *schema_sql(),
        f"DROP TABLE IF EXISTS dpu_test.{table}",
        f"CREATE TABLE dpu_test.{table} (id INTEGER PRIMARY KEY, text_col1 TEXT, text_col2 TEXT)",
    ]
    for batch in range((rows + 9) // 10):
        start_id = batch * 10
        end_id = min(start_id + 9, rows - 1)
        statements.append(
            f"""
            INSERT INTO dpu_test.{table} (id, text_col1, text_col2)
            SELECT i, repeat('x', {RAW_COLUMN_BYTES}) || i::text, repeat('y', {RAW_COLUMN_BYTES}) || i::text
            FROM generate_series({start_id}, {end_id}) AS i
            """
        )
    return tuple(statements)


def raw_fullscan_group() -> BenchmarkGroup:
    return BenchmarkGroup(
        name="read_raw_data_fullscan",
        cluster_start=70,
        cases=tuple(
            BenchmarkCase(
                f"read_raw_fullscan_{rows}rows",
                f"Read {rows} row{'s' if rows != 1 else ''} with SELECT * (full scan, no WHERE)",
                "read",
                rows * RAW_ROW_BYTES,
                f"SELECT * FROM dpu_test.read_raw_fullscan_{rows}rows",
                fetch=True,
                explain=True,
                setup_sql=fullscan_setup_sql(rows),
            )
            for rows in RAW_ROW_COUNTS
        ),
    )


def compute_group() -> BenchmarkGroup:
    return BenchmarkGroup(
        name="compute_blog",
        setup_sql=(
            *schema_sql(),
            "DROP TABLE IF EXISTS dpu_test.compute_test",
            "CREATE TABLE dpu_test.compute_test (id INTEGER, text_col TEXT, num_col INTEGER)",
            "INSERT INTO dpu_test.compute_test SELECT i, repeat('x', 100) || i::text, i * 10 FROM generate_series(0, 49) AS i",
        ),
        cases=(
            BenchmarkCase(
                "crossjoin_4way",
                "Cross join 4-way (6.25M rows) with SUM",
                "compute",
                0,
                """
                SELECT SUM(t1.id + t2.id + t3.id + t4.id)
                FROM dpu_test.compute_test t1
                CROSS JOIN dpu_test.compute_test t2
                CROSS JOIN dpu_test.compute_test t3
                CROSS JOIN dpu_test.compute_test t4
                """,
                explain=True,
            ),
            BenchmarkCase(
                "sort_3col_125k",
                "Sort 125K rows by 3 columns",
                "compute",
                0,
                """
                SELECT t1.id, t2.id, t3.id
                FROM dpu_test.compute_test t1
                CROSS JOIN dpu_test.compute_test t2
                CROSS JOIN dpu_test.compute_test t3
                ORDER BY t1.id, t2.id, t3.id
                """,
                fetch=True,
                explain=True,
            ),
            BenchmarkCase(
                "regex_simple",
                "Simple regex pattern (50 rows)",
                "compute",
                0,
                "SELECT id, text_col FROM dpu_test.compute_test WHERE text_col ~ '^[A-Za-z0-9]'",
                fetch=True,
                explain=True,
            ),
        ),
    )


def blog_groups(selection: str = "all") -> tuple[BenchmarkGroup, ...]:
    groups = {
        "writes": (write_group("no_pk"), write_group("serial_pk"), write_group("uuid_pk")),
        "single-reads": (single_read_group("no_pk"), single_read_group("serial_pk"), single_read_group("uuid_pk")),
        "raw-where": (raw_where_group(),),
        "raw-fullscan": (raw_fullscan_group(),),
        "compute": (compute_group(),),
    }
    if selection == "all":
        return tuple(group for key in ["writes", "single-reads", "raw-where", "raw-fullscan", "compute"] for group in groups[key])
    return groups[selection]


def render_group_sql(group: BenchmarkGroup) -> str:
    lines = [f"-- {group.name}", ""]
    if group.setup_sql:
        lines.append("-- setup")
        lines.extend(statement.strip() + ";" for statement in group.setup_sql)
        lines.append("")
    for case in group.cases:
        lines.append(f"-- {case.test_id}: {case.description}")
        if case.setup_sql:
            lines.append("-- case setup")
            lines.extend(statement.strip() + ";" for statement in case.setup_sql)
        lines.append(case.sql.strip() + ";")
        lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["all", "writes", "single-reads", "raw-where", "raw-fullscan", "compute"], default="all")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sql = "\n\n".join(render_group_sql(group) for group in blog_groups(args.only))
    print(sql)


if __name__ == "__main__":
    main()
