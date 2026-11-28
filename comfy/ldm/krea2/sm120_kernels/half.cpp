#include <cublasLt.h>
#include <windows.h>
#include <memory>
static decltype(&cublasLtCreate) pcublasLtCreate = nullptr;
static decltype(&cublasLtMatmul) pcublasLtMatmul = nullptr;
static decltype(&cublasLtMatmulAlgoGetHeuristic) pcublasLtMatmulAlgoGetHeuristic = nullptr;
static decltype(&cublasLtMatmulDescCreate) pcublasLtMatmulDescCreate = nullptr;
static decltype(&cublasLtMatmulDescSetAttribute) pcublasLtMatmulDescSetAttribute = nullptr;
static decltype(&cublasLtMatmulPreferenceCreate) pcublasLtMatmulPreferenceCreate = nullptr;
static decltype(&cublasLtMatmulPreferenceSetAttribute) pcublasLtMatmulPreferenceSetAttribute = nullptr;
static decltype(&cublasLtMatrixLayoutCreate) pcublasLtMatrixLayoutCreate = nullptr;
static decltype(&cublasLtMatmulPreferenceDestroy) pcublasLtMatmulPreferenceDestroy = nullptr;
static decltype(&cublasLtMatrixLayoutDestroy) pcublasLtMatrixLayoutDestroy = nullptr;
static decltype(&cublasLtMatmulDescDestroy) pcublasLtMatmulDescDestroy = nullptr;
static decltype(&cublasLtDestroy) pcublasLtDestroy = nullptr;
extern "C" __declspec(dllexport) int configure(const wchar_t* file) {
  auto lib = LoadLibraryExW(file, nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
  if (!lib) return -int(GetLastError());
  pcublasLtCreate = reinterpret_cast<decltype(&cublasLtCreate)>(GetProcAddress(lib, "cublasLtCreate"));
  if (!pcublasLtCreate) return -1000-int(GetLastError());
  pcublasLtMatmul = reinterpret_cast<decltype(&cublasLtMatmul)>(GetProcAddress(lib, "cublasLtMatmul"));
  if (!pcublasLtMatmul) return -1000-int(GetLastError());
  pcublasLtMatmulAlgoGetHeuristic = reinterpret_cast<decltype(&cublasLtMatmulAlgoGetHeuristic)>(GetProcAddress(lib, "cublasLtMatmulAlgoGetHeuristic"));
  if (!pcublasLtMatmulAlgoGetHeuristic) return -1000-int(GetLastError());
  pcublasLtMatmulDescCreate = reinterpret_cast<decltype(&cublasLtMatmulDescCreate)>(GetProcAddress(lib, "cublasLtMatmulDescCreate"));
  if (!pcublasLtMatmulDescCreate) return -1000-int(GetLastError());
  pcublasLtMatmulDescSetAttribute = reinterpret_cast<decltype(&cublasLtMatmulDescSetAttribute)>(GetProcAddress(lib, "cublasLtMatmulDescSetAttribute"));
  if (!pcublasLtMatmulDescSetAttribute) return -1000-int(GetLastError());
  pcublasLtMatmulPreferenceCreate = reinterpret_cast<decltype(&cublasLtMatmulPreferenceCreate)>(GetProcAddress(lib, "cublasLtMatmulPreferenceCreate"));
  if (!pcublasLtMatmulPreferenceCreate) return -1000-int(GetLastError());
  pcublasLtMatmulPreferenceSetAttribute = reinterpret_cast<decltype(&cublasLtMatmulPreferenceSetAttribute)>(GetProcAddress(lib, "cublasLtMatmulPreferenceSetAttribute"));
  if (!pcublasLtMatmulPreferenceSetAttribute) return -1000-int(GetLastError());
  pcublasLtMatrixLayoutCreate = reinterpret_cast<decltype(&cublasLtMatrixLayoutCreate)>(GetProcAddress(lib, "cublasLtMatrixLayoutCreate"));
  if (!pcublasLtMatrixLayoutCreate) return -1000-int(GetLastError());
  pcublasLtMatmulPreferenceDestroy = reinterpret_cast<decltype(&cublasLtMatmulPreferenceDestroy)>(GetProcAddress(lib, "cublasLtMatmulPreferenceDestroy"));
  if (!pcublasLtMatmulPreferenceDestroy) return -1000-int(GetLastError());
  pcublasLtMatrixLayoutDestroy = reinterpret_cast<decltype(&cublasLtMatrixLayoutDestroy)>(GetProcAddress(lib, "cublasLtMatrixLayoutDestroy"));
  if (!pcublasLtMatrixLayoutDestroy) return -1000-int(GetLastError());
  pcublasLtMatmulDescDestroy = reinterpret_cast<decltype(&cublasLtMatmulDescDestroy)>(GetProcAddress(lib, "cublasLtMatmulDescDestroy"));
  if (!pcublasLtMatmulDescDestroy) return -1000-int(GetLastError());
  pcublasLtDestroy = reinterpret_cast<decltype(&cublasLtDestroy)>(GetProcAddress(lib, "cublasLtDestroy"));
  if (!pcublasLtDestroy) return -1000-int(GetLastError());
  return 0;
}
#include <cublasLt.h>
#include <cuda_fp16.h>
struct Search {
 cublasLtHandle_t h{}; cublasLtMatmulDesc_t op{};
 cublasLtMatrixLayout_t a{},b{},c{}; cublasLtMatmulPreference_t p{};
 cublasLtMatmulHeuristicResult_t choices[32]{}; int count{};
 ~Search(){
  if (p) pcublasLtMatmulPreferenceDestroy(p);
  if (a) pcublasLtMatrixLayoutDestroy(a);
  if (b) pcublasLtMatrixLayoutDestroy(b);
  if (c) pcublasLtMatrixLayoutDestroy(c);
  if (op) pcublasLtMatmulDescDestroy(op);
  if (h) pcublasLtDestroy(h);
 }

};
#define OK(x) if(auto status=(x);status!=CUBLAS_STATUS_SUCCESS) return -int(status)
extern "C" __declspec(dllexport) int prepare(Search** ptr,int m,int n,int k,size_t bytes){
 auto s=std::make_unique<Search>(); cublasOperation_t t=CUBLAS_OP_T;
 OK(pcublasLtCreate(&s->h));OK(pcublasLtMatmulDescCreate(&s->op,CUBLAS_COMPUTE_16F,CUDA_R_16F));
 OK(pcublasLtMatmulDescSetAttribute(s->op,CUBLASLT_MATMUL_DESC_TRANSA,&t,sizeof(t)));
 OK(pcublasLtMatrixLayoutCreate(&s->a,CUDA_R_16F,k,n,k));
 OK(pcublasLtMatrixLayoutCreate(&s->b,CUDA_R_16F,k,m,k));
 OK(pcublasLtMatrixLayoutCreate(&s->c,CUDA_R_16F,n,m,n));
 OK(pcublasLtMatmulPreferenceCreate(&s->p));
 OK(pcublasLtMatmulPreferenceSetAttribute(s->p,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&bytes,sizeof(bytes)));
 OK(pcublasLtMatmulAlgoGetHeuristic(s->h,s->op,s->a,s->b,s->c,s->c,s->p,32,s->choices,&s->count));
 int count=s->count;
 if(count)*ptr=s.release();
 return count;
}
extern "C" __declspec(dllexport) int run(Search* s,int i,void* a,void* w,void* out,void* work,size_t bytes,void* stream){
 unsigned short alpha=0x3c00,beta=0;
 OK(pcublasLtMatmul(s->h,s->op,&alpha,w,s->a,a,s->b,&beta,out,s->c,out,s->c,&s->choices[i].algo,work,bytes,(cudaStream_t)stream));
 return 0;
}

extern "C" __declspec(dllexport) void destroy_half(Search* s){delete s;}
