"""Aurora DSQL CloudWatch metric collection helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import boto3

DEFAULT_METRICS = [
    "TotalDPU",
    "ReadDPU",
    "WriteDPU",
    "ComputeDPU",
    "MultiRegionWriteDPU",
    "BytesRead",
    "BytesWritten",
    "ComputeTime",
    "CommitLatency",
    "TotalTransactions",
    "ReadOnlyTransactions",
    "OccConflicts",
    "QueryTimeouts",
]
DEFAULT_PERIOD_SECONDS = 60
DEFAULT_STATISTIC = "Sum"


def parse_time(value: str) -> datetime:
    value = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def collect_one_metric(
    cloudwatch: Any,
    cluster_id: str,
    metric_name: str,
    start_time: datetime,
    end_time: datetime,
    period: int,
    statistic: str,
    dimension_names: list[str],
) -> dict[str, Any]:
    last_response: dict[str, Any] | None = None

    for dimension_name in dimension_names:
        response = cloudwatch.get_metric_statistics(
            Namespace="AWS/AuroraDSQL",
            MetricName=metric_name,
            Dimensions=[{"Name": dimension_name, "Value": cluster_id}],
            StartTime=start_time,
            EndTime=end_time,
            Period=period,
            Statistics=[statistic],
        )
        last_response = response
        datapoints = response.get("Datapoints", [])
        if datapoints:
            datapoints = sorted(datapoints, key=lambda item: item["Timestamp"])
            total = sum(point.get(statistic, 0.0) for point in datapoints)
            return {
                "dimension_name": dimension_name,
                "statistic": statistic,
                "total": total,
                "datapoints": len(datapoints),
                "details": [
                    {
                        "timestamp": point["Timestamp"].astimezone(timezone.utc).isoformat(),
                        statistic: point.get(statistic),
                    }
                    for point in datapoints
                ],
            }

    return {
        "dimension_name": dimension_names[0],
        "statistic": statistic,
        "total": 0.0,
        "datapoints": 0,
        "details": [],
        "raw_checked": bool(last_response is not None),
    }


def collect_metrics(
    cluster_id: str,
    region: str | None,
    start_time: datetime,
    end_time: datetime,
    metrics: list[str] | None = None,
    period: int = 60,
    statistic: str = "Sum",
    dimension_name: str | None = None,
) -> dict[str, Any]:
    session = boto3.Session()
    cloudwatch = session.client("cloudwatch", region_name=region)
    effective_region = cloudwatch.meta.region_name
    dimension_names = [dimension_name] if dimension_name else ["ResourceId", "ClusterId"]
    metric_names = metrics or DEFAULT_METRICS

    collected = {
        "cluster_id": cluster_id,
        "region": effective_region,
        "start_time": start_time.astimezone(timezone.utc).isoformat(),
        "end_time": end_time.astimezone(timezone.utc).isoformat(),
        "period_seconds": period,
        "metrics": {},
    }

    for metric_name in metric_names:
        collected["metrics"][metric_name] = collect_one_metric(
            cloudwatch=cloudwatch,
            cluster_id=cluster_id,
            metric_name=metric_name,
            start_time=start_time,
            end_time=end_time,
            period=period,
            statistic=statistic,
            dimension_names=dimension_names,
        )

    return collected


def summarize(collected: dict[str, Any]) -> dict[str, float]:
    return {
        metric_name: metric_data.get("total", 0.0)
        for metric_name, metric_data in collected.get("metrics", {}).items()
    }
