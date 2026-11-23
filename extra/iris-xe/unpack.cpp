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

static __m256 factor(float d, int scale) {
    return _mm256_set1_ps(_cvtsh_ss(_cvtss_sh(d * scale, 0)));
}

static __m256i field(const uint8_t* p, int shift, int mask) {
    return _mm256_and_si256(_mm256_srli_epi32(load_eight(p), shift), _mm256_set1_epi32(mask));
}

static void affine(float* dst, __m256 scale, __m256i q, __m256 minimum) {
    auto product = round_half(_mm256_mul_ps(scale, _mm256_cvtepi32_ps(q)));
    _mm256_storeu_ps(dst, round_half(_mm256_sub_ps(product, minimum)));
}

static void decode_q5(const uint8_t* p, float* dst) {
    for (int g = 0; g < 8; ++g) {
        int j = g % 4;
        int s = p[4+j], m = p[8+j], sm = p[12+j];
        auto scale = factor(read_half(p), g < 4 ? s & 63 : (sm & 15) | ((s >> 6) << 4));
        auto minimum = factor(read_half(p+2), g < 4 ? m & 63 : (sm >> 4) | ((m >> 6) << 4));
        for (int i = 0; i < 32; i += 8) {
            auto low = field(p+48+(g/2)*32+i, (g%2)*4, 15);
            auto high = field(p+16+i, g, 1);
            affine(dst+g*32+i, scale, _mm256_or_si256(low, _mm256_slli_epi32(high,4)), minimum);
        }
    }
}

static void decode_q8(const uint8_t* p, float* dst) {
    auto d = _mm256_set1_ps(read_half(p));
    for (int i = 0; i < 32; i += 8) {
        auto q = _mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p+2+i)));
        _mm256_storeu_ps(dst+i, round_half(_mm256_mul_ps(d, _mm256_cvtepi32_ps(q))));
    }
}

static void decode_q2(const uint8_t* p, float* dst) {
    for (int g = 0; g < 16; ++g) {
        auto scale = factor(read_half(p+80), p[g] & 15);
        auto minimum = factor(read_half(p+82), p[g] >> 4);
        for (int i = 0; i < 16; i += 8) {
            auto q = field(p+16+(g/8)*32+(g%2)*16+i, (g/2%4)*2, 3);
            affine(dst+g*16+i, scale, q, minimum);
        }
    }
}

static void decode_q3(const uint8_t* p, float* dst) {
    for (int g = 0; g < 16; ++g) {
        int low = (p[96+g%8] >> (g/8*4)) & 15;
        int high = (p[104+g%4] >> (g/4*2)) & 3;
        auto scale = factor(read_half(p+108), (low | (high << 4)) - 32);
        for (int i = 0; i < 16; i += 8) {
            int lane = g%2*16+i;
            auto lo = field(p+32+(g/8)*32+lane, (g/2%4)*2, 3);
            auto high_bit = field(p+lane, g/2, 1);
            auto sign = _mm256_slli_epi32(_mm256_xor_si256(high_bit, _mm256_set1_epi32(1)), 2);
            auto q = _mm256_sub_epi32(lo, sign);
            _mm256_storeu_ps(dst+g*16+i, round_half(_mm256_mul_ps(scale, _mm256_cvtepi32_ps(q))));
        }
    }
}

template<int packed_size, int elements, void (*decode)(const uint8_t*, float*)>
static void blocks(const uint8_t* p, float* dst, int64_t count, int threads) {
    #pragma omp parallel for schedule(static) num_threads(threads) if(count >= 256)
    for (int64_t i = 0; i < count; ++i)
        decode(p+i*packed_size, dst+i*elements);
}

extern "C" int decode_mixed_fp32(int ggml_type, const uint8_t* p, float* dst, int64_t count, int threads) {
    switch (ggml_type) {
    case 10: // GGML_TYPE_Q2_K
        blocks<84,256,decode_q2>(p,dst,count,threads); break;
    case 11: // GGML_TYPE_Q3_K
        blocks<110,256,decode_q3>(p,dst,count,threads); break;
    case 13: // GGML_TYPE_Q5_K
        blocks<176,256,decode_q5>(p,dst,count,threads); break;
    case 8: // GGML_TYPE_Q8_0
        blocks<34,32,decode_q8>(p,dst,count,threads); break;
    default: return -1;
    }
    return 0;
}

#pragma GCC pop_options
