#!/usr/bin/env python3
"""Measure seekdb vector-index build timing: CREATE INDEX + first query."""
import argparse, os, subprocess, sys, time
import numpy as np
import pymysql

SEEKDB = os.environ.get("SEEKDB_HOME", "/seekdb_workspace") + "/bin/seekdb"


def start_observer(base, port=2881):
    os.makedirs(base, exist_ok=True)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = "%s/bridge:/opt/hipvs/lib:/opt/rocm/lib" % os.environ.get("SEEKDB_HOME", "/seekdb_workspace")
    log = open(base + ".out", "w")
    subprocess.Popen([SEEKDB, "--base-dir=" + base, "--port=%d" % port, "--nodaemon",
                      "--parameter", "_memory_budget=16G", "--parameter", "datafile_size=20G",
                      "--parameter", "datafile_maxsize=40G", "--parameter", "log_disk_size=10G",
                      "--parameter", "_max_syslog_file_count=10",
                      "--log-level", "INFO"],
                     stdout=log, stderr=log, env=env, cwd=base)
    for _ in range(120):
        try:
            c = pymysql.connect(host="127.0.0.1", port=port, user="root", password="", autocommit=True)
            c.cursor().execute("select 1")
            return c
        except Exception:
            time.sleep(2)
    raise RuntimeError("observer did not start")


def lit(row):
    return "[" + ",".join("%.6f" % x for x in row) + "]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--lib", default="cuvs")
    ap.add_argument("--port", type=int, default=2881)
    ap.add_argument("--base", default="/root/work/obs1")
    ap.add_argument("--rows-per-insert", type=int, default=1000)
    ap.add_argument("--skip-start", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(42)
    base_vecs = rng.standard_normal((args.n, args.dim)).astype(np.float32)
    queries = rng.standard_normal((10, args.dim)).astype(np.float32)

    if args.skip_start:
        conn = pymysql.connect(host="127.0.0.1", port=args.port, user="root", password="", autocommit=True)
        conn.cursor().execute("select 1")
    else:
        conn = start_observer(args.base, args.port)
    cur = conn.cursor()
    cur.execute("select version()")
    print("version:", cur.fetchone()[0], flush=True)

    cur.execute("create database if not exists bench")
    cur.execute("use bench")
    cur.execute("alter system set ob_vector_memory_limit_percentage=30")
    cur.execute("drop table if exists t")
    cur.execute("create table t(id int primary key, v vector(%d))" % args.dim)

    t_load = time.time()
    for s in range(0, args.n, args.rows_per_insert):
        e = min(s + args.rows_per_insert, args.n)
        sql = "insert into t values " + ",".join(
            "(%d,'%s')" % (i, lit(base_vecs[i])) for i in range(s, e))
        cur.execute(sql)
    print("load_s=%.3f" % (time.time() - t_load), flush=True)

    # ---- timed: create index ----
    t0 = time.time()
    cur.execute("create vector index idx on t(v) with (distance=l2, type=hnsw, lib=%s)" % args.lib)
    t1 = time.time()
    print("create_s=%.3f" % (t1 - t0), flush=True)

    # ---- timed: first query ----
    q = lit(queries[0])
    cur.execute("select id from t order by l2_distance(v,'%s') approximate limit 10" % q)
    rows = cur.fetchall()
    t2 = time.time()
    print("first_query_s=%.3f" % (t2 - t1), flush=True)
    print("e2e_s=%.3f" % (t2 - t0), flush=True)
    print("first_ids=%s" % ([r[0] for r in rows],), flush=True)

    lat = []
    for i in range(10):
        qq = lit(queries[i % 10])
        tt = time.time()
        cur.execute("select id from t order by l2_distance(v,'%s') approximate limit 10" % qq)
        cur.fetchall()
        lat.append(time.time() - tt)
    print("query_ms=%s" % ([round(x * 1000, 2) for x in lat],), flush=True)
    print("query_p50_ms=%.2f" % (1000 * float(np.percentile(lat, 50))), flush=True)


if __name__ == "__main__":
    main()
