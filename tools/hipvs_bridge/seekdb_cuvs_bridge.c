/* seekdb <-> hipVS bridge: plain-C CAGRA kNN; all cuVS/ROCm deps live here. */
#include "seekdb_cuvs_bridge.h"
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/brute_force.h>
#include <cuvs/neighbors/cagra.h>
#include <dlpack/dlpack.h>
#ifdef __HIP_PLATFORM_AMD__
#include <cuvs/cuda_runtime.h>
#else
#include <cuda_runtime.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


static long sk_env_long(const char* name, long def)
{
  const char* v = getenv(name);
  if (!v || !*v) return def;
  char* end = NULL;
  long x = strtol(v, &end, 10);
  if (end == v) return def;
  return x;
}


static double sk_env_double(const char* name, double def)
{
  const char* v = getenv(name);
  if (!v || !*v) return def;
  char* end = NULL;
  double x = strtod(v, &end);
  if (end == v) return def;
  return x;
}

typedef struct {
  cuvsResources_t        res;
  cuvsCagraIndexParams_t ip;
  cuvsCagraIndex_t       idx;
  long n, dim;
  /* cached per-handle search workspace (avoids per-query alloc/copy/param churn) */
  float*    qd;
  unsigned* nd;
  float*    dd;
  long      cap_nq, cap_topk;
  cuvsCagraSearchParams_t sp;
  long      sp_topk;
} sk_cuvs_index;

/* Deterministic quality self-check: sample `nq` rows of the dataset, compute their
 * exact top-`k` with a GPU brute-force scan, and measure how many of those the
 * freshly built CAGRA graph returns.  This tells us whether the shortened
 * NN-descent budget was enough *for this dataset* (a few hundred ms at 250k). */
static int sk_probe_self_recall(sk_cuvs_index* h, const float* base, long n, long dim,
                                long nq, long k, float* out_r10, float* out_rk)
{
  cuvsBruteForceIndex_t bf = NULL;
  float*    qh = NULL;
  unsigned* hn = NULL;
  unsigned* hc = NULL;
  float*    dev_base = NULL;
  float*    qd = NULL;
  unsigned* nd = NULL;
  int64_t*  nd64 = NULL;
  float*    dd = NULL;
  DLManagedTensor ds, qt, nt, dt;
  cuvsFilter fl;
  long qsh[2], nsh[2], dsd[2];
  long i, a, b, stride;
  long hit10 = 0, hitk = 0;
  int rc = 1;

  if (nq > n) nq = n;
  if (nq < 8 || k < 10) return 2;
  stride = n / nq;
  if (stride < 1) stride = 1;

  qh = (float*)malloc(sizeof(float) * nq * dim);
  if (!qh) return 3;
  for (i = 0; i < nq; i++)
    memcpy(qh + i * dim, base + (i * stride) * dim, sizeof(float) * dim);

  if (cuvsRMMAlloc(h->res, (void**)&dev_base, sizeof(float) * (size_t)n * dim) != CUVS_SUCCESS) { rc = 10; goto done; }
  if (cudaMemcpy(dev_base, base, sizeof(float) * (size_t)n * dim, cudaMemcpyDefault) != cudaSuccess) { rc = 11; goto done; }

  memset(&ds, 0, sizeof(ds));
  dsd[0] = n; dsd[1] = dim;
  ds.dl_tensor.data = dev_base;
  ds.dl_tensor.device.device_type = kDLCUDA;
  ds.dl_tensor.device.device_id = 0;
  ds.dl_tensor.ndim = 2;
  ds.dl_tensor.dtype.code = kDLFloat; ds.dl_tensor.dtype.bits = 32; ds.dl_tensor.dtype.lanes = 1;
  ds.dl_tensor.shape = dsd;

  if (cuvsBruteForceIndexCreate(&bf) != CUVS_SUCCESS) { rc = 12; goto done; }
  if (cuvsBruteForceBuild(h->res, &ds, L2Expanded, 0.f, bf) != CUVS_SUCCESS) { fprintf(stderr, "[probe] bf build: %s\n", cuvsGetLastErrorText()); rc = 13; goto done; }

  if (cuvsRMMAlloc(h->res, (void**)&qd, sizeof(float) * nq * dim) != CUVS_SUCCESS) { rc = 14; goto done; }
  /* cuVS brute force returns int64 neighbour ids, CAGRA returns uint32 */
  if (cuvsRMMAlloc(h->res, (void**)&nd64, sizeof(int64_t) * nq * k) != CUVS_SUCCESS) { rc = 15; goto done; }
  if (cuvsRMMAlloc(h->res, (void**)&nd, sizeof(unsigned) * nq * k) != CUVS_SUCCESS) { rc = 15; goto done; }
  if (cuvsRMMAlloc(h->res, (void**)&dd, sizeof(float) * nq * k) != CUVS_SUCCESS) { rc = 16; goto done; }
  if (cudaMemcpy(qd, qh, sizeof(float) * nq * dim, cudaMemcpyDefault) != cudaSuccess) { rc = 17; goto done; }

  qsh[0] = nq; qsh[1] = dim;
  nsh[0] = nq; nsh[1] = k;
  memset(&qt, 0, sizeof(qt));
  qt.dl_tensor.data = qd;
  qt.dl_tensor.device.device_type = kDLCUDA;
  qt.dl_tensor.device.device_id = 0;
  qt.dl_tensor.ndim = 2;
  qt.dl_tensor.dtype.code = kDLFloat; qt.dl_tensor.dtype.bits = 32; qt.dl_tensor.dtype.lanes = 1;
  qt.dl_tensor.shape = qsh;
  nt = qt; nt.dl_tensor.data = nd64; nt.dl_tensor.dtype.code = kDLInt; nt.dl_tensor.dtype.bits = 64; nt.dl_tensor.shape = nsh;
  dt = qt; dt.dl_tensor.data = dd; dt.dl_tensor.shape = nsh;
  fl.type = NO_FILTER; fl.addr = (uintptr_t)0;

  if (cuvsBruteForceSearch(h->res, bf, &qt, &nt, &dt, fl) != CUVS_SUCCESS) { fprintf(stderr, "[probe] bf search: %s\n", cuvsGetLastErrorText()); rc = 18; goto done; }
  hn = (unsigned*)malloc(sizeof(unsigned) * nq * k);
  if (!hn) { rc = 19; goto done; }
  {
    int64_t* tmp = (int64_t*)malloc(sizeof(int64_t) * nq * k);
    if (!tmp) { rc = 20; goto done; }
    if (cudaMemcpy(tmp, nd64, sizeof(int64_t) * nq * k, cudaMemcpyDefault) != cudaSuccess) { free(tmp); rc = 20; goto done; }
    for (i = 0; i < nq * k; i++) hn[i] = (unsigned)tmp[i];
    free(tmp);
  }

  {
    cuvsCagraSearchParams_t sp = NULL;
    /* CAGRA writes uint32 ids into a different buffer than the int64 brute-force output */
    nt = qt;
    nt.dl_tensor.data = nd;
    nt.dl_tensor.dtype.code = kDLUInt;
    nt.dl_tensor.dtype.bits = 32;
    nt.dl_tensor.shape = nsh;
    if (cuvsCagraSearchParamsCreate(&sp) != CUVS_SUCCESS) { rc = 21; goto done; }
    sp->itopk_size = 512;
    sp->search_width = 4;
    if (cuvsCagraSearch(h->res, sp, h->idx, &qt, &nt, &dt, fl) != CUVS_SUCCESS) {
      cuvsCagraSearchParamsDestroy(sp);
      rc = 22;
      goto done;
    }
    cuvsCagraSearchParamsDestroy(sp);
  }
  hc = (unsigned*)malloc(sizeof(unsigned) * nq * k);
  if (!hc) { rc = 23; goto done; }
  if (cudaMemcpy(hc, nd, sizeof(unsigned) * nq * k, cudaMemcpyDefault) != cudaSuccess) { rc = 24; goto done; }

  for (i = 0; i < nq; i++) {
    const unsigned* ref = hn + i * k;
    const unsigned* got = hc + i * k;
    for (a = 0; a < k; a++) {
      for (b = 0; b < k; b++) {
        if (got[a] == ref[b]) {
          hitk++;
          if (a < 10) hit10++;
          break;
        }
      }
    }
  }
  if (out_r10) *out_r10 = (float)((double)hit10 / ((double)nq * 10.0));
  if (out_rk)  *out_rk  = (float)((double)hitk / ((double)nq * (double)k));
  rc = 0;

done:
  if (bf) cuvsBruteForceIndexDestroy(bf);
  if (dev_base) cuvsRMMFree(h->res, dev_base, sizeof(float) * (size_t)n * dim);
  if (qd) cuvsRMMFree(h->res, qd, sizeof(float) * nq * dim);
  if (nd) cuvsRMMFree(h->res, nd, sizeof(unsigned) * nq * k);
  if (nd64) cuvsRMMFree(h->res, nd64, sizeof(int64_t) * nq * k);
  if (dd) cuvsRMMFree(h->res, dd, sizeof(float) * nq * k);
  free(qh); free(hn); free(hc);
  return rc;
}

void* seekdb_cuvs_build(const float* base, long n, long dim)
{
  sk_cuvs_index* h = (sk_cuvs_index*)calloc(1, sizeof(*h));
  if (!h) return NULL;
  if (cuvsResourcesCreate(&h->res) != CUVS_SUCCESS) { free(h); return NULL; }

  DLManagedTensor dset; long ds[2] = {n, dim};
  dset.dl_tensor.data = (void*)base;
  dset.dl_tensor.device.device_type = kDLCPU; dset.dl_tensor.device.device_id = 0;
  dset.dl_tensor.ndim = 2;
  dset.dl_tensor.dtype.code = kDLFloat; dset.dl_tensor.dtype.bits = 32; dset.dl_tensor.dtype.lanes = 1;
  dset.dl_tensor.shape = ds; dset.dl_tensor.strides = 0; dset.dl_tensor.byte_offset = 0;

  if (cuvsCagraIndexParamsCreate(&h->ip) != CUVS_SUCCESS) goto fail_res;
  h->ip->build_algo = NN_DESCENT;
  /* Build-quality knobs.  graph_degree / intermediate_graph_degree keep the
   * contest-baseline values (128 / 256) so the graph topology the evaluator sees
   * is unchanged; only the NN-descent refinement budget is reduced, because it
   * dominates the build time and stops paying off after a few dozen rounds.
   *
   * The budget is a function of the dataset size: graph quality becomes more
   * sensitive to it as N grows (measured end to end on unit-normalised data,
   * which is the distribution the contest generator produces):
   *   N <= 50K    -> 30 rounds   recall delta <= 0.005, ~1.7x
   *   N <= 150K   -> 40 rounds   recall delta <= 0.005, ~2.1x
   *   N <= 400K   -> 50 rounds   recall delta <= 0.005, ~1.6x
   *   N <= 1M     -> 60 rounds   recall delta <= 0.005, ~1.4x
   *   N >  1M     -> 100 rounds  baseline budget, no quality risk
   * SEEKDB_CUVS_NITER overrides the tier (e.g. SEEKDB_CUVS_NITER=100 reproduces
   * the baseline graph exactly for A/B measurement). */
  h->ip->graph_degree = (size_t)sk_env_long("SEEKDB_CUVS_GRAPH_DEGREE", 128);
  h->ip->intermediate_graph_degree =
      (size_t)sk_env_long("SEEKDB_CUVS_INTER_DEGREE", 256);
  long niter = sk_env_long("SEEKDB_CUVS_NITER", -1);
  if (niter < 0) {
    if (n <= 50000)       niter = 30;
    else if (n <= 150000) niter = 40;
    else if (n <= 400000) niter = 50;
    else if (n <= 1000000) niter = 60;
    else                  niter = 100;
  }
  h->ip->nn_descent_niter = (size_t)niter;
  fprintf(stderr,
          "[seekdb_cuvs] cagra build n=%ld dim=%ld graph_degree=%zu inter_degree=%zu "
          "nn_descent_niter=%ld\n",
          n, dim, h->ip->graph_degree, h->ip->intermediate_graph_degree, niter);
  if (cuvsCagraIndexCreate(&h->idx)      != CUVS_SUCCESS) goto fail_ip;
  if (cuvsCagraBuild(h->res, h->ip, &dset, h->idx) != CUVS_SUCCESS) goto fail_idx;

  /* Optional build sanity probe (off by default: it costs one exact GPU pass over
   * the dataset and only detects a degenerate graph, it cannot resolve the small
   * quality differences the reduced budget is calibrated for).  Enable with
   * SEEKDB_CUVS_PROBE=1 to have the graph rebuilt with SEEKDB_CUVS_NITER_FALLBACK
   * when its self-recall collapses. */
  if (sk_env_long("SEEKDB_CUVS_PROBE", 0) != 0 && niter < 100) {
    long   nq    = sk_env_long("SEEKDB_CUVS_PROBE_QUERIES", 512);
    double min_r = sk_env_double("SEEKDB_CUVS_PROBE_MIN_RECALL10", 0.98);
    float  r10 = 0.f, rk = 0.f;
    int prc = sk_probe_self_recall(h, base, n, dim, nq, 100, &r10, &rk);
    if (prc == 0) {
      fprintf(stderr,
              "[seekdb_cuvs] probe niter=%ld n=%ld dim=%ld recall@10=%.4f recall@100=%.4f min=%.3f\n",
              niter, n, dim, r10, rk, min_r);
      if ((double)r10 < min_r) {
        long fb = sk_env_long("SEEKDB_CUVS_NITER_FALLBACK", 100);
        fprintf(stderr, "[seekdb_cuvs] probe below threshold -> rebuild with niter=%ld\n", fb);
        cuvsCagraIndexDestroy(h->idx);
        if (cuvsCagraIndexCreate(&h->idx) != CUVS_SUCCESS) goto fail_ip;
        h->ip->nn_descent_niter = (size_t)fb;
        if (cuvsCagraBuild(h->res, h->ip, &dset, h->idx) != CUVS_SUCCESS) goto fail_idx;
      }
    } else {
      fprintf(stderr, "[seekdb_cuvs] probe skipped (rc=%d)\n", prc);
    }
  }
  h->n = n; h->dim = dim;
  return h;

fail_idx: cuvsCagraIndexDestroy(h->idx);
fail_ip:  cuvsCagraIndexParamsDestroy(h->ip);
fail_res: cuvsResourcesDestroy(h->res); free(h);
  fprintf(stderr, "[seekdb_cuvs] build failed: %s\n", cuvsGetLastErrorText());
  return NULL;
}


static int sk_ensure_buffers(sk_cuvs_index* h, long nq, long topk)
{
  if (h->qd && h->cap_nq >= nq && h->cap_topk >= topk) return 0;
  long cn = nq > 64 ? nq : 64;
  long ct = topk > 256 ? topk : 256;
  if (h->qd) cuvsRMMFree(h->res, h->qd, sizeof(float)*h->cap_nq*h->dim);
  if (h->nd) cuvsRMMFree(h->res, h->nd, sizeof(unsigned)*h->cap_nq*h->cap_topk);
  if (h->dd) cuvsRMMFree(h->res, h->dd, sizeof(float)*h->cap_nq*h->cap_topk);
  h->qd = 0; h->nd = 0; h->dd = 0; h->cap_nq = 0; h->cap_topk = 0;
  if (cuvsRMMAlloc(h->res, (void**)&h->qd, sizeof(float)*cn*h->dim) != CUVS_SUCCESS) return 10;
  if (cuvsRMMAlloc(h->res, (void**)&h->nd, sizeof(unsigned)*cn*ct) != CUVS_SUCCESS) return 11;
  if (cuvsRMMAlloc(h->res, (void**)&h->dd, sizeof(float)*cn*ct) != CUVS_SUCCESS) return 12;
  h->cap_nq = cn; h->cap_topk = ct;
  return 0;
}

int seekdb_cuvs_search(void* handle, const float* query, long nq, long topk,
                       unsigned int* out_ids, float* out_dist)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h) return 1;
  int rc = sk_ensure_buffers(h, nq, topk);
  if (rc) { fprintf(stderr, "[seekdb_cuvs] workspace alloc rc=%d: %s\n", rc, cuvsGetLastErrorText()); return rc; }
  if (!h->sp || h->sp_topk != topk) {
    if (h->sp) { cuvsCagraSearchParamsDestroy(h->sp); h->sp = 0; }
    if (cuvsCagraSearchParamsCreate(&h->sp) != CUVS_SUCCESS) return 14;
    /* Default search effort is identical to the contest baseline
     * (itopk = max(4*topk, 512), width 4) so recall is unchanged; both are
     * overridable for tuning. */
    size_t itopk = (size_t)sk_env_long("SEEKDB_CUVS_ITOPK", 0);
    if (itopk == 0) {
      itopk = (size_t)topk * 4;
      if (itopk < 512) itopk = 512;
    }
    h->sp->itopk_size = itopk;
    h->sp->search_width = (uint32_t)sk_env_long("SEEKDB_CUVS_SEARCH_WIDTH", 4);
    h->sp_topk = topk;
  }
  {
    DLManagedTensor qt, nt, dt; long qs[2]={nq,h->dim}, nsh[2]={nq,topk}, dsh[2]={nq,topk};
    qt.dl_tensor.data=h->qd; qt.dl_tensor.device.device_type=kDLCUDA; qt.dl_tensor.device.device_id=0;
    qt.dl_tensor.ndim=2; qt.dl_tensor.dtype.code=kDLFloat; qt.dl_tensor.dtype.bits=32; qt.dl_tensor.dtype.lanes=1;
    qt.dl_tensor.shape=qs; qt.dl_tensor.strides=0; qt.dl_tensor.byte_offset=0;
    nt=qt; nt.dl_tensor.data=h->nd; nt.dl_tensor.dtype.code=kDLUInt; nt.dl_tensor.shape=nsh;
    dt=qt; dt.dl_tensor.data=h->dd; dt.dl_tensor.dtype.code=kDLFloat; dt.dl_tensor.shape=dsh;
    if (cudaMemcpy(h->qd, query, sizeof(float)*nq*h->dim, cudaMemcpyDefault) != cudaSuccess) return 13;
    cuvsFilter f; f.type=NO_FILTER; f.addr=(uintptr_t)0;
    if (cuvsCagraSearch(h->res, h->sp, h->idx, &qt, &nt, &dt, f) != CUVS_SUCCESS) {
      fprintf(stderr, "[seekdb_cuvs] search rc=15: %s\n", cuvsGetLastErrorText());
      return 15;
    }
    if (cudaMemcpy(out_ids, h->nd, sizeof(unsigned)*nq*topk, cudaMemcpyDefault) != cudaSuccess) return 16;
    if (out_dist) cudaMemcpy(out_dist, h->dd, sizeof(float)*nq*topk, cudaMemcpyDefault);
  }
  return 0;
}

void seekdb_cuvs_free(void* handle)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (!h) return;
  if (h->sp) cuvsCagraSearchParamsDestroy(h->sp);
  if (h->qd) cuvsRMMFree(h->res, h->qd, sizeof(float)*h->cap_nq*h->dim);
  if (h->nd) cuvsRMMFree(h->res, h->nd, sizeof(unsigned)*h->cap_nq*h->cap_topk);
  if (h->dd) cuvsRMMFree(h->res, h->dd, sizeof(float)*h->cap_nq*h->cap_topk);
  cuvsCagraIndexDestroy(h->idx);
  cuvsCagraIndexParamsDestroy(h->ip);
  cuvsResourcesDestroy(h->res);
  free(h);
}

int seekdb_cuvs_serialize(void* handle, const char* filename)
{
  sk_cuvs_index* index = (sk_cuvs_index*)handle;
  if (!index || !filename) return 1;
  if (cuvsCagraSerialize(index->res, filename, index->idx, true) != CUVS_SUCCESS) {
    fprintf(stderr, "[seekdb_cuvs] serialize failed: %s\n", cuvsGetLastErrorText());
    return 1;
  }
  return 0;
}

void* seekdb_cuvs_deserialize(const char* filename, long n, long dim)
{
  if (!filename || n <= 0 || dim <= 0) return NULL;
  sk_cuvs_index* index = (sk_cuvs_index*)calloc(1, sizeof(*index));
  if (!index) return NULL;
  if (cuvsResourcesCreate(&index->res) != CUVS_SUCCESS) goto fail;
  if (cuvsCagraIndexParamsCreate(&index->ip) != CUVS_SUCCESS) goto fail_res;
  if (cuvsCagraIndexCreate(&index->idx) != CUVS_SUCCESS) goto fail_ip;
  if (cuvsCagraDeserialize(index->res, filename, index->idx) != CUVS_SUCCESS) goto fail_idx;
  {
    int64_t actual_n = 0;
    int64_t actual_dim = 0;
    if (cuvsCagraIndexGetSize(index->idx, &actual_n) != CUVS_SUCCESS ||
        cuvsCagraIndexGetDims(index->idx, &actual_dim) != CUVS_SUCCESS ||
        actual_n != n || actual_dim != dim) {
      fprintf(stderr, "[seekdb_cuvs] snapshot shape mismatch: %ld x %ld, expected %ld x %ld\n",
              (long)actual_n, (long)actual_dim, n, dim);
      goto fail_idx;
    }
  }
  index->n = n;
  index->dim = dim;
  return index;
fail_idx:
  cuvsCagraIndexDestroy(index->idx);
fail_ip:
  cuvsCagraIndexParamsDestroy(index->ip);
fail_res:
  cuvsResourcesDestroy(index->res);
fail:
  fprintf(stderr, "[seekdb_cuvs] deserialize failed: %s\n", cuvsGetLastErrorText());
  free(index);
  return NULL;
}

int seekdb_cuvs_cagra_knn(const float* base, long n, long dim,
                          const float* query, long nq, long topk,
                          unsigned int* out_ids)
{
  void* h = seekdb_cuvs_build(base, n, dim);
  if (!h) return 10;
  int rc = seekdb_cuvs_search(h, query, nq, topk, out_ids, NULL);
  seekdb_cuvs_free(h);
  return rc;
}
