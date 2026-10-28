#!/usr/bin/env python3
"""seekdb x hipVS contest benchmark: build time, query latency and recall.

Every round starts from an empty base directory, loads the same deterministic
vector set, then measures:
  create_s      CREATE VECTOR INDEX (includes the full CAGRA build)
  first_query_s the mandatory first query, which must be part of the build cost
  e2e_s         create_s + first_query_s
  p50/p95       single-query latency over --queries vectors
  qps           queries per second over that batch
  recall@10/100 compared against an exact numpy ground truth on the same data
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np
import pymysql

SEEKDB = os.environ.get("SEEKDB_HOME", "/seekdb_workspace") + "/bin/seekdb"


def start_observer(base, port):
    os.makedirs(base, exist_ok=True)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = "%s/bridge:/opt/hipvs/lib:/opt/rocm/lib" % os.environ.get(
        "SEEKDB_HOME", "/seekdb_workspace")
    log = open(base + ".out", "w")
    subprocess.Popen(
        [SEEKDB, "--base-dir=" + base, "--port=%d" % port, "--nodaemon",
         "--parameter", "_memory_budget=16G", "--parameter", "datafile_size=20G",
         "--parameter", "datafile_maxsize=40G", "--parameter", "log_disk_size=10G",
         "--parameter", "_max_syslog_file_count=10", "--log-level", "INFO"],
        stdout=log, stderr=log, env=env, cwd=base)
    for _ in range(180):
        try:
            conn = pymysql.connect(host="127.0.0.1", port=port, user="root", password="",
                                   autocommit=True)
            conn.cursor().execute("select 1")
            return conn
        except Exception:
            time.sleep(2)
    raise RuntimeError("observer did not start")


def lit(row):
    return "[" + ",".join("%.6f" % x for x in row) + "]"


def ground_truth(base_vecs, queries, k):
    out = []
    for q in queries:
        d = ((base_vecs - q) ** 2).sum(axis=1)
        out.append(np.argsort(d, kind="stable")[:k])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--lib", default="cuvs")
    ap.add_argument("--port", type=int, default=2881)
    ap.add_argument("--base", default="/root/work/run/eval")
    ap.add_argument("--rows-per-insert", type=int, default=2000)
    ap.add_argument("--queries", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="")
    ap.add_argument("--norm", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    base_vecs = rng.standard_normal((args.n, args.dim)).astype(np.float32)
    queries = rng.standard_normal((args.queries, args.dim)).astype(np.float32)
    if args.norm:
        base_vecs /= np.linalg.norm(base_vecs, axis=1, keepdims=True)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    conn = start_observer(args.base, args.port)
    cur = conn.cursor()
    cur.execute("select version()")
    version = cur.fetchone()[0]
    cur.execute("drop database if exists bench")
    cur.execute("create database bench")
    cur.execute("use bench")
    cur.execute("alter system set ob_vector_memory_limit_percentage=30")
    cur.execute("create table t(id int primary key, v vector(%d))" % args.dim)

    t_load = time.time()
    for s in range(0, args.n, args.rows_per_insert):
        e = min(s + args.rows_per_insert, args.n)
        sql = "insert into t values " + ",".join(
            "(%d,'%s')" % (i, lit(base_vecs[i])) for i in range(s, e))
        cur.execute(sql)
    load_s = time.time() - t_load

    t0 = time.time()
    cur.execute("create vector index idx on t(v) with (distance=l2, type=hnsw, lib=%s)" % args.lib)
    t1 = time.time()
    create_s = t1 - t0

    def knn(q, limit=10):
        cur.execute("select id from t order by l2_distance(v,'%s') approximate limit %d"
                    % (lit(q), limit))
        return [r[0] for r in cur.fetchall()]

    first = knn(queries[0])
    t2 = time.time()
    first_query_s = t2 - t1

    lat, rows10, rows100 = [], [], []
    for i in range(args.queries):
        q = queries[i]
        tt = time.time()
        got10 = knn(q, 10)
        lat.append(time.time() - tt)
        rows10.append(got10)
        rows100.append(knn(q, 100))

    gt10 = ground_truth(base_vecs, queries[:args.queries], 10)
    gt100 = ground_truth(base_vecs, queries[:args.queries], 100)
    rec10 = float(np.mean([len(set(a) & set(int(x) for x in b)) / 10.0
                           for a, b in zip(rows10, gt10)]))
    rec100 = float(np.mean([len(set(a) & set(int(x) for x in b)) / 100.0
                            for a, b in zip(rows100, gt100)]))
    lat_ms = [x * 1000.0 for x in lat]
    result = {
        "version": version, "n": args.n, "dim": args.dim, "lib": args.lib,
        "load_s": round(load_s, 3), "create_s": round(create_s, 3),
        "first_query_s": round(first_query_s, 3),
        "e2e_s": round(create_s + first_query_s, 3),
        "first_ids": [int(x) for x in first],
        "query_p50_ms": round(float(np.percentile(lat_ms, 50)), 3),
        "query_p95_ms": round(float(np.percentile(lat_ms, 95)), 3),
        "qps": round(args.queries / max(sum(lat), 1e-9), 1),
        "recall@10": round(rec10, 4), "recall@100": round(rec100, 4),
        "empty_results": sum(1 for r in rows10 if not r),
    }
    print(json.dumps(result), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
