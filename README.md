# Aurora DSQL DPU Lab

I tried to understand what Aurora DSQL DPUs actually mean in practice.

AWS says DPUs are the unit for all the database work. Reads, writes, compute, multi-region writes. That is true, but it also does not help much when you want to know if your query costs nothing or suddenly costs real money.

So I ran a bunch of queries with some scripts and found some intersting tidbits, which are published in [a blog post](https://blog.faridnsh.ninja/unraveling-aurora-dsql-pricing).

## What Is In Here?

- `scripts/run_benchmarks.py`: orchestrate the benchmarks in parallel into a new timestamped result directory.
- `scripts/dsql_client.py`: provides reusable DSQL connection and query execution helpers.
- `scripts/real_world_workload.py`: runs the indexed write/read workload used as a rough real-world shape check.
- `scripts/plot_write_limits.py`: make the write-limit charts from enriched JSONL result files.
- `scripts/plot_read_queries.py`: make the read-query charts from enriched JSONL result files.
- `scripts/plot_signaldeck_invocations.py`: make the real-world invocation-shape chart used in the blog.
- `results/`: contains the two runs of the benchmarks.
- `assets/`: contains the various charts that I used for the blog post and maybe more.
