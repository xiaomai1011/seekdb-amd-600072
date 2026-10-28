#include "seekdb_cuvs_bridge.h"
#include <cuvs/core/c_api.h>
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
#include <time.h>

static double now_ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec/1e6;}

int main(int argc,char**argv){
  /* data n dim query nq topk degree inter niter algo out */
  const char*data=argv[1]; long n=atol(argv[2]); long dim=atol(argv[3]);
  const char*query=argv[4]; long nq=atol(argv[5]); long topk=atol(argv[6]);
  int degree=atoi(argv[7]), inter=atoi(argv[8]), niter=atoi(argv[9]), algo=atoi(argv[10]);
  size_t itopk=atoi(argv[11]); int width=atoi(argv[12]);
  const char*out=argv[13];
  int use_fp16=(argc>14)?atoi(argv[14]):0;
  float*base=(float*)malloc(sizeof(float)*n*dim);
  FILE*f=fopen(data,"rb"); if(!f||fread(base,sizeof(float),n*dim,f)!=(size_t)n*dim){printf("read base fail\n");return 1;} fclose(f);
  float*qs=(float*)malloc(sizeof(float)*nq*dim);
  f=fopen(query,"rb"); if(fread(qs,sizeof(float),nq*dim,f)!=(size_t)nq*dim){printf("read q fail\n");return 1;} fclose(f);
  cuvsResources_t res; cuvsResourcesCreate(&res);
  cuvsCagraIndexParams_t ip; cuvsCagraIndexParamsCreate(&ip);
  ip->build_algo = (algo==1)? IVF_PQ : NN_DESCENT;
  ip->graph_degree = degree; ip->intermediate_graph_degree = inter; ip->nn_descent_niter = niter;
  cuvsCagraIndex_t idx; cuvsCagraIndexCreate(&idx);
  unsigned short*base16=NULL; if(use_fp16){ base16=(unsigned short*)malloc(sizeof(unsigned short)*n*dim); for(long i=0;i<n*dim;i++){ _Float16 h=(_Float16)base[i]; memcpy(&base16[i],&h,2);} }
  DLManagedTensor ds; long dsh[2]={n,dim};
  ds.dl_tensor.data=use_fp16?(void*)base16:(void*)base; ds.dl_tensor.device.device_type=kDLCPU; ds.dl_tensor.device.device_id=0;
  ds.dl_tensor.ndim=2; ds.dl_tensor.dtype.code=kDLFloat; ds.dl_tensor.dtype.bits=use_fp16?16:32; ds.dl_tensor.dtype.lanes=1;
  ds.dl_tensor.shape=dsh; ds.dl_tensor.strides=0; ds.dl_tensor.byte_offset=0;
  double t0=now_ms();
  if(cuvsCagraBuild(res,ip,&ds,idx)!=CUVS_SUCCESS){printf("build fail: %s\n",cuvsGetLastErrorText());return 1;}
  double build=now_ms()-t0;
  float*qd; unsigned*nd; float*dd;
  cuvsRMMAlloc(res,(void**)&qd,(use_fp16?2:4)*nq*dim);
  cuvsRMMAlloc(res,(void**)&nd,sizeof(unsigned)*nq*topk);
  cuvsRMMAlloc(res,(void**)&dd,sizeof(float)*nq*topk);
  if(use_fp16){ unsigned short*q16=(unsigned short*)malloc(sizeof(unsigned short)*nq*dim); for(long i=0;i<nq*dim;i++){ _Float16 h=(_Float16)qs[i]; memcpy(&q16[i],&h,2);} cudaMemcpy(qd,q16,sizeof(unsigned short)*nq*dim,cudaMemcpyDefault);} else cudaMemcpy(qd,qs,sizeof(float)*nq*dim,cudaMemcpyDefault);
  cuvsCagraSearchParams_t sp; cuvsCagraSearchParamsCreate(&sp);
  size_t need=(size_t)topk*4; if(need<512) need=512; if(itopk>need) need=itopk; sp->itopk_size=need; sp->search_width=width;
  double t1=now_ms();
  /* batched search, 256 queries per call to bound memory */
  unsigned*ids=(unsigned*)malloc(sizeof(unsigned)*nq*topk);
  for(long i=0;i<nq;i+=256){
    long m=(nq-i<256)?(nq-i):256;
    DLManagedTensor qt,nt,dt; long qsh[2]={m,dim}, nsh[2]={m,topk}, ddh[2]={m,topk};
    qt.dl_tensor.data=qd+ (size_t)i*dim; qt.dl_tensor.device.device_type=kDLCUDA; qt.dl_tensor.device.device_id=0;
    qt.dl_tensor.ndim=2; qt.dl_tensor.dtype.code=kDLFloat; qt.dl_tensor.dtype.bits=use_fp16?16:32; qt.dl_tensor.dtype.lanes=1;
    qt.dl_tensor.shape=qsh; qt.dl_tensor.strides=0; qt.dl_tensor.byte_offset=0;
    nt=qt; nt.dl_tensor.data=nd; nt.dl_tensor.dtype.code=kDLUInt; nt.dl_tensor.dtype.bits=32; nt.dl_tensor.shape=nsh;
    dt=qt; dt.dl_tensor.data=dd; dt.dl_tensor.dtype.code=kDLFloat; dt.dl_tensor.dtype.bits=32; dt.dl_tensor.shape=ddh;
    cuvsFilter fl; fl.type=NO_FILTER; fl.addr=(uintptr_t)0;
    if(cuvsCagraSearch(res,sp,idx,&qt,&nt,&dt,fl)!=CUVS_SUCCESS){printf("search fail: %s\n",cuvsGetLastErrorText());return 1;}
    cudaMemcpy(ids+(size_t)i*topk,nd,sizeof(unsigned)*m*topk,cudaMemcpyDefault);
  }
  double search=now_ms()-t1;
  FILE*o=fopen(out,"wb"); fwrite(ids,sizeof(unsigned),nq*topk,o); fclose(o);
  printf("{\"build_ms\":%.1f,\"search_ms_total\":%.1f,\"per_query_ms\":%.3f,\"degree\":%d,\"inter\":%d,\"niter\":%d,\"algo\":%d,\"itopk\":%zu,\"width\":%d,\"n\":%ld,\"dim\":%ld}\n",
         build,search,search/nq,degree,inter,niter,algo,need,width,n,dim);
  return 0;
}
