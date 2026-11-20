// CPU decoder for the default ComfyUI-GGUF FP16 arithmetic.
// Expand to FP32 as we write the result, for an FP32 Iris Xe denoiser.
#include <cstdint>
#include <cstring>
#include <immintrin.h>

extern "C" bool iris_unpack_supported() {
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c");
}

#pragma GCC push_options
#pragma GCC target("avx2,f16c")

static float read_half(const uint8_t* ptr) {
    uint16_t bits;
    std::memcpy(&bits, ptr, sizeof(bits));
    return _cvtsh_ss(bits);
}

static float round_half(float value) {
    return _cvtsh_ss(_cvtss_sh(value, 0));
}

static __m256 round_half(__m256 value) {
    return _mm256_cvtph_ps(_mm256_cvtps_ph(value, 0));
}

static __m256i load_eight(const uint8_t* ptr) {
    return _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(ptr)));
}

static void decode_q4(const uint8_t* p, float* out) {
    const uint8_t* scales = p + 4;
    const uint8_t* packed = p + 16;
    for (int group = 0; group < 8; ++group) {
        const int s = group < 4 ? scales[group] & 63 : (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4);
        const int m = group < 4 ? scales[group + 4] & 63 : (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4);
        const __m256 factor = _mm256_set1_ps(round_half(read_half(p) * s));
        const __m256 minimum = _mm256_set1_ps(round_half(read_half(p + 2) * m));
        for (int j = 0; j < 32; j += 8) {
            const __m256i bytes = load_eight(packed + (group / 2) * 32 + j);
            const __m256i q = _mm256_and_si256(_mm256_srli_epi32(bytes, (group % 2) * 4), _mm256_set1_epi32(15));
            const __m256 product = round_half(_mm256_mul_ps(factor, _mm256_cvtepi32_ps(q)));
            _mm256_storeu_ps(out + group * 32 + j, round_half(_mm256_sub_ps(product, minimum)));
        }
    }
}

static void decode_q6(const uint8_t* p, float* out) {
    for (int group = 0; group < 8; ++group) {
        const int section = group / 4;
        const int lane = group % 4;
        for (int j = 0; j < 32; j += 8) {
            const __m256i low = load_eight(p + section * 64 + (lane % 2) * 32 + j);
            const __m256i high = load_eight(p + 128 + section * 32 + j);
            const __m256i lo = _mm256_and_si256(_mm256_srli_epi32(low, (lane / 2) * 4), _mm256_set1_epi32(15));
            const __m256i hi = _mm256_and_si256(_mm256_srli_epi32(high, lane * 2), _mm256_set1_epi32(3));
            const __m256i q = _mm256_sub_epi32(_mm256_or_si256(lo, _mm256_slli_epi32(hi, 4)), _mm256_set1_epi32(32));
            const int8_t scale = static_cast<int8_t>(p[192 + group * 2 + j / 16]);
            const __m256 factor = _mm256_set1_ps(round_half(read_half(p + 208) * scale));
            _mm256_storeu_ps(out + group * 32 + j, round_half(_mm256_mul_ps(factor, _mm256_cvtepi32_ps(q))));
        }
    }
}

extern "C" void decode_fp32(const uint8_t* packed, float* out, int64_t blocks, bool q4, int threads) {
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (int64_t b = 0; b < blocks; ++b) {
        if (q4) decode_q4(packed + 144 * b, out + 256 * b);
        else decode_q6(packed + 210 * b, out + 256 * b);
    }
}

#pragma GCC pop_options
