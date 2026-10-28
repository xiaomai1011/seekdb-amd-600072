import argparse, json, os, random, socket, subprocess, time
import numpy as np, pymysql

def wait_port(port, timeout=300):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = socket.socket()
        s.settimeout(1)
        try:
            s.connect(("127.0.0.1", port)); s.close(); return True
        except Exception:
            time.sleep(0.5)
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250000)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--port", type=int, default=3101)
    ap.add_argument("--base", default="/root/work/run/v5_250k")
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    base = rng.random((a.n, a.dim), dtype=np.float32)
    qry = rng.random((a.queries, a.dim), dtype=np.float32)
    env = dict(os.environ); env["LD_LIBRARY_PATH"] = "/root/work/v5_home/bridge"
    proc = subprocess.Popen(["/root/work/v5_home/bin/seekdb", a.base, "--port", str(a.port)],
                            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        assert wait_port(a.port), "server did not start"
        time.sleep(3)
        conn = pymysql.connect(host="127.0.0.1", port=a.port, user="root", password="", database="bench", autocommit=True)
        cur = conn.cursor()
        def lit(v):
            return "[" + ",".join("%.6f" % x for x in v) + "]"
        cur.execute("select id from t order by id limit 1"); first_id = cur.fetchone()[0]
        lat = []
        empty = 0
        for i in range(a.queries):
            q = qry[i] + base[0]
            t0 = time.perf_counter()
            cur.execute("select id from t order by l2_distance(v, %s) approximate limit 10", (lit(q),))
            rows = cur.fetchall()
            lat.append((time.perf_counter() - t0) * 1000)
            if not rows: empty += 1
        lat = np.array(lat)
        out = {"n": a.n, "dim": a.dim, "queries": a.queries,
               "p50_ms": round(float(np.percentile(lat, 50)), 3),
               "p95_ms": round(float(np.percentile(lat, 95)), 3),
               "qps": round(1000.0 / float(np.mean(lat)), 1),
               "empty": empty}
        print(json.dumps(out))
        conn.close()
    finally:
        proc.terminate()
        try: proc.wait(timeout=20)
        except Exception: proc.kill()

if __name__ == "__main__":
    main()
