#include <cublasLt.h>
#include <memory>

struct Plan {
    cublasLtHandle_t handle = nullptr;
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, d = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulHeuristicResult_t algorithm = {};
    bool has_bias = false;

    ~Plan() {
        if (pref) cublasLtMatmulPreferenceDestroy(pref);
        if (a) cublasLtMatrixLayoutDestroy(a);
        if (b) cublasLtMatrixLayoutDestroy(b);
        if (d) cublasLtMatrixLayoutDestroy(d);
        if (op) cublasLtMatmulDescDestroy(op);
        if (handle) cublasLtDestroy(handle);
    }

    template<typename T>
    cublasStatus_t set(cublasLtMatmulDescAttributes_t attr, T value) {
        return cublasLtMatmulDescSetAttribute(op, attr, &value, sizeof(value));
    }
};

#define CHECK(call) do { auto status = (call); if (status != CUBLAS_STATUS_SUCCESS) return static_cast<int>(status); } while (false)

extern "C" __declspec(dllexport) int prepare_mx(
    void** result, uint64_t m, uint64_t n, uint64_t k, int bf16,
    const void* all_one_scale_a, const void* all_one_scale_b,
    const void* bias, size_t workspace_size, int use_device_scalars
) {
    auto plan = std::make_unique<Plan>();
    CHECK(cublasLtCreate(&plan->handle));
    CHECK(cublasLtMatmulDescCreate(&plan->op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    CHECK(plan->set(CUBLASLT_MATMUL_DESC_TRANSA, CUBLAS_OP_T));
    CHECK(plan->set(CUBLASLT_MATMUL_DESC_TRANSB, CUBLAS_OP_N));
    if (use_device_scalars) {
        CHECK(plan->set(CUBLASLT_MATMUL_DESC_POINTER_MODE, CUBLASLT_POINTER_MODE_DEVICE));
    }
    // Column-major N x M is the same storage as the row-major M x N output.
    CHECK(plan->set(CUBLASLT_MATMUL_DESC_A_SCALE_MODE, CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0));
    CHECK(plan->set(CUBLASLT_MATMUL_DESC_B_SCALE_MODE, CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0));
    CHECK(plan->set(CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, all_one_scale_b));
    CHECK(plan->set(CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, all_one_scale_a));
    cudaDataType_t out_type = bf16 ? CUDA_R_16BF : CUDA_R_16F;
    plan->has_bias = bias != nullptr;
    if (bias) {
        CHECK(plan->set(CUBLASLT_MATMUL_DESC_EPILOGUE, CUBLASLT_EPILOGUE_BIAS));
        CHECK(plan->set(CUBLASLT_MATMUL_DESC_BIAS_POINTER, bias));
        CHECK(plan->set(CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, out_type));
    }
    CHECK(cublasLtMatrixLayoutCreate(&plan->a, CUDA_R_8F_E4M3, k, n, k));
    CHECK(cublasLtMatrixLayoutCreate(&plan->b, CUDA_R_8F_E4M3, k, m, k));
    CHECK(cublasLtMatrixLayoutCreate(&plan->d, out_type, n, m, n));
    CHECK(cublasLtMatmulPreferenceCreate(&plan->pref));
    CHECK(cublasLtMatmulPreferenceSetAttribute(
        plan->pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)
    ));
    int found = 0;
    CHECK(cublasLtMatmulAlgoGetHeuristic(
        plan->handle, plan->op, plan->a, plan->b, plan->d, plan->d,
        plan->pref, 1, &plan->algorithm, &found
    ));
    if (found == 0) return -1;
    *result = plan.release();
    return 0;
}

extern "C" __declspec(dllexport) int run_mx(
    Plan* plan, const void* input, const void* weight, const void* alpha, const void* beta,
    const void* bias, void* output, void* workspace, size_t workspace_size, void* stream
) {
    if (plan->has_bias) CHECK(plan->set(CUBLASLT_MATMUL_DESC_BIAS_POINTER, bias));
    return static_cast<int>(cublasLtMatmul(
        plan->handle, plan->op, alpha,
        weight, plan->a, input, plan->b, beta, output, plan->d, output, plan->d,
        &plan->algorithm.algo, workspace, workspace_size, static_cast<cudaStream_t>(stream)
    ));
}

extern "C" __declspec(dllexport) void destroy_mx(Plan* plan) {
    delete plan;
}
