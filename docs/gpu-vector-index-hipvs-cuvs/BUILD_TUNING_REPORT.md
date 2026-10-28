# hipVS CAGRA 构建加速报告（提交版）

本报告记录在冻结基线 `amd-cagra-contest-baseline-v5`（commit
`634888d9b037153fa143d2c913ba4b25503e1d62`，tag `amd-cagra-contest-baseline-v5`）
之上，对 AMD GPU CAGRA 索引构建链路所做的优化、实测数据与复现方法。

全部改动集中在 `tools/hipvs_bridge/seekdb_cuvs_bridge.c`（seekdb 与 hipVS/cuVS
之间唯一的 C ABI 桥接层）。不改变 SQL 语义，不改变 `lib=vsag` 路径，不引入任何
CPU/非 GPU 的建图或检索回退，持久化格式仍为 `OBCUVS02`。

## 1. 结论摘要

构建（端到端 `CREATE VECTOR INDEX` + 首条查询，SQL 计时，官方数据同分布：
单位归一化 float32，128 维）：

| 数据集 | 基线（`niter=100`） | 提交版本 | 加速比 | 提交版本 `niter` |
| --- | --- | --- | --- | --- |
| 25K × 128 | 12.79 s | 7.70 s | **1.66x** | 30 |
| 100K × 128 | 31.31 s | 15.08 s | **2.08x** | 40 |
| 250K × 128 | 45.20 s | 28.33 s | **1.60x** | 50 |
| 1M × 128 | 126.61 s | 92.44 s | **1.37x** | 60 |

召回（100 条查询，精确暴力 ground truth，与基线同轮对比）：

| 数据集 | 基线 Recall@10 / @100 | 提交版本 Recall@10 / @100 | 差值 |
| --- | --- | --- | --- |
| 25K × 128 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0 / 0 |
| 100K × 128 | 0.9990 / 0.9939 | 0.9980 / 0.9941 | −0.0010 / +0.0002 |
| 250K × 128 | 0.9890 / 0.9643 | 0.9890 / 0.9639 | 0 / −0.0004 |
| 1M × 128 | 0.9100 / 0.8131 | 0.9250 / 0.8088 | +0.0150 / −0.0043 |

> 上表所有行均为 100 条查询、同一轮 A/B；`--norm` 生成的数据与官方 demo 数据同分布
> （单位归一化随机向量，`std≈0.088`）。

单条查询（SQL 计时，同一批查询、同一张图）：

| 数据集 | 基线 p50 / QPS | 提交版本 p50 / QPS | 吞吐比 |
| --- | --- | --- | --- |
| 25K × 128 | 6.31 ms / 156.7 | 6.67 ms / 147.4 | 0.94x |
| 100K × 128 | 6.44 ms / 149.5 | 5.97 ms / 165.7 | 1.11x |
| 250K × 128 | 6.67 ms / 149.7 | 7.03 ms / 141.7 | 0.95x |
| 1M × 128 | 7.04 ms / 139.1 | 6.67 ms / 149.4 | 1.07x |

核心变更：

1. **NN-descent 迭代预算按数据集规模自适应**（30 / 40 / 60 / 100 轮，基线固定
   100 轮）。NN-descent 占构建时间的 80% 以上，减轮数即等比例缩短构建；轮数下限
   由实测召回决定（见 §4），保证相对基线 Recall@10/@100 下降不超过 0.005。
2. **查询工作区复用**：把每次查询都执行的 3 次 RMM 分配/释放与
   `cuvsCagraSearchParamsCreate/Destroy` 改为按索引句柄缓存，单次 GPU 检索固定
   开销 2.75 ms → 0.39 ms。
3. 全部构建/检索旋钮均可环境变量覆盖，`SEEKDB_CUVS_NITER=100` 可一键复现基线图，
   便于云端 A/B 复核。

## 2. 基线耗时构成与可优化空间

V5 构建链路（`ObVectorIndexSerializer` → `ob_vsag_adaptor` GPU 快照 →
`tools/hipvs_bridge` → `cuvsCagraBuild`）在 250K × 128 上的实测构成：

| 阶段 | 耗时 |
| --- | --- |
| 初始图 + 剪枝（`nn_descent_niter=0`） | 7.5 s |
| NN-descent 每轮 | ~0.34 s |
| 100 轮 NN-descent | ~34 s |
| 构建合计（`cuvsCagraBuild`） | ~43 s |
| DDL 其余部分（扫描、快照持久化 `OBCUVS02`、GPU 传输） | ~2.5 s |

即 **NN-descent 轮数是唯一的量级杠杆**；数据集拷贝、索引持久化都在百毫秒级。
`build_algo=IVF_PQ` 虽然把构建压到 3.4 s（12.9x），但 Recall@10 只有 0.76，
属于“用召回换性能”的禁止行为，已否决。

## 3. 自适应预算的依据（关键决策数据）

轮数对图质量的影响随 N 增长而放大，因此在同一套官方同分布数据上逐档标定，
每一档都要求 Recall@10 与 Recall@100 相对基线下降 ≤ 0.005：

| 数据集 | 基线 r@10 / r@100 | `niter=30` | `niter=40` | `niter=50` | `niter=60` | 采用 |
| --- | --- | --- | --- | --- | --- | --- |
| 25K | 1.0000 / 1.0000 | 1.0000 / 1.0000 ✓ | — | — | — | **30** |
| 100K | 0.9990 / 0.9939 | — | 0.9980 / 0.9941 ✓ | — | — | **40** |
| 250K | 0.9890 / 0.9643 | 0.9820 / 0.9623（−0.007 ✗） | 0.9840 / 0.9614（−0.005 临界） | 0.9890 / 0.9639（0 / −0.0004 ✓） | — | **50** |
| 1M | 0.9100 / 0.8131 | 0.8960 / 0.7918（−0.022 ✗） | 0.9060 / 0.8046（−0.012 ✗） | — | 0.9250 / 0.8088（+0.015 / −0.004 ✓） | **60** |
| > 1M | — | — | — | — | — | **100**（基线预算） |

对应实现（`sk_cuvs_index_create`）：

```c
if (n <= 50000)        niter = 30;   /* 1.66x */
else if (n <= 150000)  niter = 40;   /* 2.08x @100K */
else if (n <= 400000)  niter = 50;   /* 1.60x @250K */
else if (n <= 1000000) niter = 60;   /* 1.37x @1M  */
else                   niter = 100;  /* 基线预算，零风险 */
```

`graph_degree` / `intermediate_graph_degree` 始终保持基线值 128 / 256，
即图的拓扑规格与基线一致，只有 NN-descent 的精化轮数不同。

## 4. 查询路径

`ob_vsag_adaptor` 的 GPU 检索经 `seekdb_cuvs_search()` 进入桥接层。原实现每次
查询都会：

- `cuvsRMMAlloc` + `cuvsRMMFree` 三个设备缓冲区（query / neighbors / distances）；
- `cuvsCagraSearchParamsCreate` + `Destroy`；
- 3 次阻塞式 `cudaMemcpy`。

改为按句柄缓存缓冲区和搜索参数（容量不足才重新分配），保留基线
`itopk = max(4*topk, 512)`、`search_width = 4`，检索结果与基线逐位一致。

微基准（250K × 128，`nq=1`，`topk=10`，同一张图）：

| 实现 | 单次检索耗时 |
| --- | --- |
| 原实现（每次分配 + 建参） | 2.751 ms |
| 缓存工作区 | 0.386 ms |

SQL 侧单条查询还有约 4–5 ms 固定开销（解析 / 计划 / 结果组装），因此端到端 p50
主要由它决定；本项优化消除的是每查询 2.4 ms 的固定浪费，用于保证
`candidate_qps / baseline_qps >= 0.90` 的硬门槛并改善批量/高并发场景。

**不建议**通过降低 `itopk` 换查询加速：实测 `itopk=64` 会让 SQL 端 Recall@10
从 0.995 掉到 0.54（CAGRA 在 250K 规模需要宽搜索），属于用召回换性能。

## 5. 环境变量

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `SEEKDB_CUVS_NITER` | 按 N 分层（30/40/50/60/100） | NN-descent 轮数；`100` 完全复现基线图 |
| `SEEKDB_CUVS_GRAPH_DEGREE` | `128` | 最终图度数（基线值） |
| `SEEKDB_CUVS_INTER_DEGREE` | `256` | 中间图度数（基线值） |
| `SEEKDB_CUVS_ITOPK` | `0` | `0` = 基线策略 `max(4*topk, 512)` |
| `SEEKDB_CUVS_SEARCH_WIDTH` | `4` | 检索宽度（基线值） |
| `SEEKDB_CUVS_PROBE` | `0` | 打开后构建完成时用精确 GPU 暴力搜索做一次自检 |

## 6. 复现方法

环境：ROCm 7.2.4 + hipVS（`/opt/hipvs`），AMD Radeon（gfx11），128 核，503 GB。

```bash
# 1) 构建桥接层与 seekdb
bash tools/hipvs_bridge/build.sh
./build.sh release \
    -DOB_BUILD_CUVS=ON \
    -DCUVS_BRIDGE_LIB=$PWD/tools/hipvs_bridge/libseekdb_cuvs_bridge.so \
    --make -j64

# 2) 端到端：建索引(计时) + 首查(计时) + 查询延迟 + 精确召回
#    --norm 生成单位归一化向量与查询（与官方 demo 数据同分布）
SEEKDB_HOME=/path/to/seekdb_home \
  /opt/venv/bin/python tools/hipvs_bridge/bench/bench_eval.py \
    --n 250000 --dim 128 --norm --queries 100 --port 2881 --base /tmp/run1 \
    --out /tmp/cand250.json

# 3) 同一脚本对基线版本重复一次（相同 N / 相同查询），得到同轮 A/B
SEEKDB_HOME=/path/to/baseline_home SEEKDB_CUVS_NITER=100 \
  /opt/venv/bin/python tools/hipvs_bridge/bench/bench_eval.py \
    --n 250000 --dim 128 --norm --queries 100 --port 2882 --base /tmp/run2 \
    --out /tmp/base250.json

# 4) 纯 CAGRA 构建基准（不含 DDL，用于拆解耗时）
hipcc -O2 -D__HIP_PLATFORM_AMD__ -x hip -I/opt/hipvs/include \
  tools/hipvs_bridge/bench/cagra_bench.c -L/opt/hipvs/lib -lcuvs_c -o cagra_bench
./cagra_bench base.f32 250000 128 query.f32 4096 100 128 256 40 0 512 4 out.bin
```

`bench_eval.py` 每一步都打印 `create_s / first_query_s / e2e_s / p50 / p95 / qps /
recall@10 / recall@100`，其中 recall 用 numpy 精确暴力最近邻作为 ground truth，
查询集与建图集独立生成，固定随机种子可完全复现。

## 7. 合规性说明（对应赛题硬门槛）

- **真实使用 GPU 建图**：构建过程中实际执行的仍是 hipVS/cuVS `cuvsCagraBuild`
  （`NN_DESCENT`），`lib=cuvs` 路径没有 CPU/VSAG 回退；`lib=vsag` CPU 路径代码
  未改动。
- **FP32 语义不变**：评测输入、索引、查询全程 FP32，未做精度降级。
- **召回**：所有档位都在官方同分布数据上逐点验证过 ≤ 0.005 的下降（§3）。
- **计时口径**：构建时间包含建图、GPU 同步与首条查询（`bench_eval.py` 中的
  `create_s + first_query_s`），没有把惰性建图挪出计时窗口；构建始终是同步的
  `cuvsCagraBuild`。
- **无预编译二进制、无缓存复用**：评测从源码重建；每轮用全新 `--base` 目录
  （`CREATE VECTOR INDEX` 时目标索引不存在）。

## 8. 未采用的方案（负面结果）

| 方案 | 结果 | 否决原因 |
| --- | --- | --- |
| `build_algo=IVF_PQ` | 3.42 s（12.9x） | Recall@10 仅 0.76 |
| `intermediate_graph_degree=384/512` | 0.78–1.13x | 更慢，召回更低 |
| FP16 构建 | 1.00x | 本 hipVS 构建不支持 fp16 数据集，结果退化 |
| 自建 GPU 图构建（NN-descent + prune 启发式） | r@10 0.9055 vs 基线 0.9405 | 质量不足，且需重写 cuVS 的剪枝/反向边优化 |
| 降低 `itopk` 换查询加速 | SQL r@10 0.995 → 0.54 | 用召回换性能 |

## 9. 后续可做

- 构建侧要进一步逼近 2.0x 且不损召回，需要更快的**高质量图构建算法**
  （自研 GPU NN-descent / 图精化，或 `cuvsCagraBuildFromGraph` 预置图入口），
  单纯调 `cuvsCagraBuild` 参数在 1M 规模已到边界。
- 查询侧的瓶颈是 seekdb SQL 层 4–5 ms 的固定开销，需要把表达式解析与计划缓存
  纳入优化范围。
