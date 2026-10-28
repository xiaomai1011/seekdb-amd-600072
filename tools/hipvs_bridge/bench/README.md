# CAGRA build/query measurement scripts

Scripts used for the numbers in
[`../../docs/gpu-vector-index-hipvs-cuvs/BUILD_TUNING_REPORT.md`](../../docs/gpu-vector-index-hipvs-cuvs/BUILD_TUNING_REPORT.md).

| file | purpose |
| --- | --- |
| `bench_eval.py` | end-to-end: start `seekdb`, load N×dim random vectors, time `CREATE VECTOR INDEX`, first query, single-query p50/p95/QPS and recall |
| `bench_query.py` | query-only latency on an already-built data directory (no DDL) |
| `cagra_bench.c` | standalone `cuvsCagraBuild` benchmark: build time, batch search latency, top-k ids dump |
| `sweep_bench.c` | build once, then sweep several `itopk`/`search_width`/`max_iterations` search settings |
| `make_gt_fast.py` | exact brute-force ground truth (top-100) for a `<tag>_{base,query}.f32` pair |
| `recal.py` | recall@10 / recall@100 of a dumped id file against the ground truth |

Data layout: `<tag>_base.f32` and `<tag>_query.f32` are raw little-endian float32,
row-major, `n × dim` and `nq × dim`.

```bash
hipcc -O2 -D__HIP_PLATFORM_AMD__ -x hip -I/opt/hipvs/include cagra_bench.c \
  -L/opt/hipvs/lib -Wl,-rpath,/opt/hipvs/lib:/opt/rocm/lib -lcuvs_c -o cagra_bench
./cagra_bench base.f32 250000 128 query.f32 4096 100 128 256 50 0 512 4 ids.bin
python3 make_gt_fast.py <tag> 250000 128 4096
python3 recal.py <tag> ids.bin 4096
```

Bridge knobs used by these runs: `SEEKDB_CUVS_NITER`, `SEEKDB_CUVS_GRAPH_DEGREE`,
`SEEKDB_CUVS_INTER_DEGREE`, `SEEKDB_CUVS_ITOPK`, `SEEKDB_CUVS_SEARCH_WIDTH`
(see the report for defaults).
