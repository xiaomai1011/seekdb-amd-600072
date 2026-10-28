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
  const char*bf=argv[1]; long n=atol(argv[2]); long dim=atol(argv[3]);
  const char*qf=argv[4]; long nq=atol(argv[5]); long topk=atol(argv[6]);
  long deg=atol(argv[7]), inter=atol(argv[8]), niter=atol(argv[9]);
  const char*cfgs=argv[10];
  float*base=(float*)malloc(sizeof(float)*n*dim); FILE*f=fopen(bf,"rb");
  if(fread(base,sizeof(float),n*dim,f)!=(size_t)n*dim){printf("read base fail\n");return 1;} fclose(f);
  float*qs=(float*)malloc(sizeof(float)*nq*dim); f=fopen(qf,"rb");
  if(fread(qs,sizeof(float),nq*dim,f)!=(size_t)nq*dim){printf("read q fail\n");return 1;} fclose(f);
  cuvsResources_t res; cuvsResourcesCreate(&res);
  cuvsCagraIndexParams_t ip; cuvsCagraIndexParamsCreate(&ip);
  ip->build_algo=NN_DESCENT; ip->graph_degree=deg; ip->intermediate_graph_degree=inter; ip->nn_descent_niter=niter;
  cuvsCagraIndex_t idx; cuvsCagraIndexCreate(&idx);
  DLManagedTensor ds; long dsh[2]={n,dim};
  ds.dl_tensor.data=base; ds.dl_tensor.device.device_type=kDLCPU; ds.dl_tensor.device.device_id=0;
  ds.dl_tensor.ndim=2; ds.dl_tensor.dtype.code=kDLFloat; ds.dl_tensor.dtype.bits=32; ds.dl_tensor.dtype.lanes=1;
  ds.dl_tensor.shape=dsh; ds.dl_tensor.strides=0; ds.dl_tensor.byte_offset=0;
  double t0=now_ms();
  if(cuvsCagraBuild(res,ip,&ds,idx)!=CUVS_SUCCESS){printf("build fail %s\n",cuvsGetLastErrorText());return 1;}
  fprintf(stderr,"build_ms=%.1f\n",now_ms()-t0);
  float*qd; unsigned*nd; float*dd;
  cuvsRMMAlloc(res,(void**)&qd,sizeof(float)*nq*dim);
  cuvsRMMAlloc(res,(void**)&nd,sizeof(unsigned)*nq*topk);
  cuvsRMMAlloc(res,(void**)&dd,sizeof(float)*nq*topk);
  cudaMemcpy(qd,qs,sizeof(float)*nq*dim,cudaMemcpyDefault);
  DLManagedTensor qt,nt,dt; long qsh[2]={nq,dim}, nsh[2]={nq,topk}, ddh[2]={nq,topk};
  qt.dl_tensor.data=qd; qt.dl_tensor.device.device_type=kDLCUDA; qt.dl_tensor.device.device_id=0;
  qt.dl_tensor.ndim=2; qt.dl_tensor.dtype.code=kDLFloat; qt.dl_tensor.dtype.bits=32; qt.dl_tensor.dtype.lanes=1;
  qt.dl_tensor.shape=qsh; qt.dl_tensor.strides=0; qt.dl_tensor.byte_offset=0;
  nt=qt; nt.dl_tensor.data=nd; nt.dl_tensor.dtype.code=kDLUInt; nt.dl_tensor.shape=nsh;
  dt=qt; dt.dl_tensor.data=dd; dt.dl_tensor.dtype.code=kDLFloat; dt.dl_tensor.shape=ddh;
  char*list=strdup(cfgs); char*save=NULL; int ci=0;
  for(char*tok=strtok_r(list,",",&save);tok;tok=strtok_r(NULL,",",&save)){
    long itopk=512,width=4,maxit=0,minit=0;
    sscanf(tok,"%ld:%ld:%ld:%ld",&itopk,&width,&maxit,&minit);
    cuvsCagraSearchParams_t sp; cuvsCagraSearchParamsCreate(&sp);
    sp->itopk_size=itopk; sp->search_width=width; sp->max_iterations=maxit; sp->min_iterations=minit;
    cuvsFilter fl; fl.type=NO_FILTER; fl.addr=(uintptr_t)0;
    double s0=now_ms();
    if(cuvsCagraSearch(res,sp,idx,&qt,&nt,&dt,fl)!=CUVS_SUCCESS){printf("search fail: %s\n",cuvsGetLastErrorText());return 1;}
    double el=now_ms()-s0;
    unsigned*out=(unsigned*)malloc(sizeof(unsigned)*nq*topk);
    cudaMemcpy(out,nd,sizeof(unsigned)*nq*topk,cudaMemcpyDefault);
    char path[256]; snprintf(path,sizeof(path),"/tmp/sw_%d.bin",ci);
    FILE*o=fopen(path,"wb"); fwrite(out,sizeof(unsigned),nq*topk,o); fclose(o);
    printf("cfg%d %s nq_batch_ms=%.1f per_query_ms=%.3f -> %s\n",ci,tok,el,el/nq,path);
    ci++; free(out); cuvsCagraSearchParamsDestroy(sp);
  }
  return 0;
}
