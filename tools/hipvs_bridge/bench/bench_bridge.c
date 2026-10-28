// Microbenchmark: bridge build + N single-query searches (nq=1 each).
// Optional 8th arg: ground-truth ids (uint32, nq*topk) to report recall.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include "seekdb_cuvs_bridge.h"

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
  if (argc < 8) {
    fprintf(stderr, "usage: %s base n dim query nq topk out [gt]\n", argv[0]);
    return 2;
  }
  const char *base_path = argv[1];
  long n = atol(argv[2]);
  long dim = atol(argv[3]);
  const char *query_path = argv[4];
  long nq = atol(argv[5]);
  long topk = atol(argv[6]);
  const char *out_path = argv[7];
  const char *gt_path = argc > 8 ? argv[8] : NULL;

  float *base = (float *)malloc(sizeof(float) * n * dim);
  FILE *f = fopen(base_path, "rb");
  if (!f || fread(base, sizeof(float), (size_t)n * dim, f) != (size_t)n * dim) {
    fprintf(stderr, "read base fail\n"); return 1;
  }
  fclose(f);
  float *qs = (float *)malloc(sizeof(float) * nq * dim);
  f = fopen(query_path, "rb");
  if (!f || fread(qs, sizeof(float), (size_t)nq * dim, f) != (size_t)nq * dim) {
    fprintf(stderr, "read query fail\n"); return 1;
  }
  fclose(f);
  unsigned *gt = NULL;
  if (gt_path) {
    gt = (unsigned *)malloc(sizeof(unsigned) * nq * topk);
    f = fopen(gt_path, "rb");
    if (!f || fread(gt, sizeof(unsigned), (size_t)nq * topk, f) != (size_t)nq * topk) {
      fprintf(stderr, "read gt fail\n"); return 1;
    }
    fclose(f);
  }

  double t0 = now_ms();
  void *h = seekdb_cuvs_build(base, n, dim);
  double build_ms = now_ms() - t0;
  if (h == NULL) { fprintf(stderr, "build fail\n"); return 1; }

  unsigned *all = (unsigned *)malloc(sizeof(unsigned) * nq * topk);
  float *dist = (float *)malloc(sizeof(float) * topk);
  double t1 = now_ms();
  for (long i = 0; i < nq; ++i) {
    if (seekdb_cuvs_search(h, qs + (size_t)i * dim, 1, topk, all + (size_t)i * topk, dist) != 0) {
      fprintf(stderr, "search fail at %ld\n", i); return 1;
    }
  }
  double search_ms = now_ms() - t1;
  double recall = -1.0;
  if (gt) {
    long hit = 0;
    for (long i = 0; i < nq; ++i) {
      for (long a = 0; a < topk; ++a) {
        for (long b = 0; b < topk; ++b) {
          if (all[(size_t)i * topk + a] == gt[(size_t)i * topk + b]) { ++hit; break; }
        }
      }
    }
    recall = (double)hit / ((double)nq * (double)topk);
  }
  FILE *o = fopen(out_path, "wb");
  if (o) { fwrite(all, sizeof(unsigned), (size_t)nq * topk, o); fclose(o); }
  printf("{\"build_ms\":%.1f,\"search_ms_total\":%.1f,\"per_query_ms\":%.3f,"
         "\"nq\":%ld,\"topk\":%ld,\"recall\":%.5f}\n",
         build_ms, search_ms, search_ms / nq, nq, topk, recall);
  seekdb_cuvs_free(h);
  return 0;
}
