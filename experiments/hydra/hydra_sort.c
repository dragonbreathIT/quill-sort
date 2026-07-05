/* hydra_sort.c — HydraSort v0.5: adaptive pass-skipping LSD radix sort
 * for 32/64-bit integers. A compiled backend candidate for quill-sort.
 *
 * Build:  gcc -O3 -march=native -shared -fPIC hydra_sort.c -o libhydra.so
 *
 * The "multiple formulas" design, chosen by one cheap measuring pass:
 *   Pass A (1 read): min, max, sorted?, reverse-sorted?  (flipped domain,
 *                    so signed/unsigned share one code path; the sign flip
 *                    lives in address math and never touches memory)
 *     -> presorted:       return            (O(n), free win)
 *     -> reverse-sorted:  in-place reverse  (O(n))
 *     -> constant:        return
 *   range < 2^16      -> counting sort      (histogram + reconstruct)
 *   otherwise         -> LSD radix using ONLY k = ceil(bits(max-min)/RB)
 *                        digit positions; min-subtraction makes offset data
 *                        (timestamps, IDs, enums) collapse to few passes.
 *     Pass B (1 read): fused k histograms.
 *     k scatter passes, ping-pong, software prefetch; pass 0 transforms
 *     the domain, the final pass restores it; digit width auto-picks
 *     10 or 11 bits (measured L2 sweet spot) minimizing pass count.
 *
 * Returns 0 = OK, 1 = out of memory, 2 = n too large (needs n < 2^32).
 * hydra_last_path() reports which strategy ran (for telemetry/tuning).
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define HYDRA_OK 0
#define HYDRA_ENOMEM 1
#define HYDRA_TOO_BIG 2
#ifndef PF_DIST
#define PF_DIST 32
#endif
#define SMALL_N 96
#define MAX_K 7            /* ceil(64/10) */
#define MAX_RADIX 2048     /* 11-bit */

/* Thread-local like the histograms below: quill may dispatch sorts from a thread
 * pool, so per-thread path telemetry avoids a benign-but-real data race. */
static _Thread_local const char *g_last_path = "none";
const char *hydra_last_path(void) { return g_last_path; }

static _Thread_local void *g_scratch = NULL;
static _Thread_local size_t g_scratch_cap = 0;
static void *get_scratch(size_t bytes) {
    if (g_scratch_cap < bytes) {
        free(g_scratch);
        g_scratch = malloc(bytes);
        g_scratch_cap = g_scratch ? bytes : 0;
    }
    return g_scratch;
}
void hydra_free_scratch(void) { free(g_scratch); g_scratch = NULL; g_scratch_cap = 0; }

static _Thread_local uint32_t g_hist[MAX_K][MAX_RADIX];
static _Thread_local uint32_t g_off[MAX_RADIX];
static _Thread_local uint32_t g_cnt[65536];

#define DEFINE_HYDRA(NAME, T, UT, FLIP, WIDTH)                                \
static void insertion_##NAME(T *a, int64_t n) {                               \
    for (int64_t i = 1; i < n; i++) {                                         \
        T x = a[i]; int64_t j = i - 1;                                        \
        while (j >= 0 && a[j] > x) { a[j+1] = a[j]; j--; }                    \
        a[j+1] = x;                                                           \
    }                                                                         \
}                                                                             \
int hydra_sort_##NAME(T *a, int64_t n) {                                      \
    if (n < 2) { g_last_path = "trivial"; return HYDRA_OK; }                  \
    if (n < SMALL_N) {                                                        \
        insertion_##NAME(a, n); g_last_path = "insertion"; return HYDRA_OK;   \
    }                                                                         \
    if (n > 0xFFFFFFFFll) return HYDRA_TOO_BIG;                               \
    UT f0 = (UT)a[0] ^ (UT)FLIP;                                              \
    UT mn = f0, mx = f0, prev = f0;                                           \
    int up = 1, down = 1;                                                     \
    for (int64_t i = 1; i < n; i++) {                                         \
        UT f = (UT)a[i] ^ (UT)FLIP;                                           \
        mn = f < mn ? f : mn;                                                 \
        mx = f > mx ? f : mx;                                                 \
        up   &= (f >= prev);                                                  \
        down &= (f <= prev);                                                  \
        prev = f;                                                             \
    }                                                                         \
    if (up)      { g_last_path = "presorted"; return HYDRA_OK; }              \
    if (mn == mx){ g_last_path = "constant";  return HYDRA_OK; }              \
    if (down) {                                                               \
        for (int64_t i = 0, j = n-1; i < j; i++, j--) {                       \
            T t = a[i]; a[i] = a[j]; a[j] = t;                                \
        }                                                                     \
        g_last_path = "reversed"; return HYDRA_OK;                            \
    }                                                                         \
    UT range = mx - mn;                                                       \
    if (range < 65536u) {                                                     \
        memset(g_cnt, 0, ((size_t)range + 1) * sizeof(uint32_t));             \
        for (int64_t i = 0; i < n; i++)                                       \
            g_cnt[(uint32_t)(((UT)a[i] ^ (UT)FLIP) - mn)]++;                  \
        T *out = a;                                                           \
        for (uint32_t v = 0; v <= (uint32_t)range; v++) {                     \
            uint32_t c = g_cnt[v]; T val = (T)((mn + (UT)v) ^ (UT)FLIP);      \
            while (c--) *out++ = val;                                         \
        }                                                                     \
        g_last_path = "counting"; return HYDRA_OK;                            \
    }                                                                         \
    int msb = 0; { UT r = range; while (r) { msb++; r >>= 1; } }              \
    int rb = 11;                                                              \
    { int k10 = (msb + 9) / 10, k11 = (msb + 10) / 11;                        \
      if (k10 == k11) rb = 10; }                                              \
    const uint32_t RMASK = (1u << rb) - 1u;                                   \
    const int k = (msb + rb - 1) / rb;                                        \
    memset(g_hist, 0, (size_t)k * MAX_RADIX * sizeof(uint32_t));              \
    for (int64_t i = 0; i < n; i++) {                                         \
        UT f = ((UT)a[i] ^ (UT)FLIP) - mn;                                    \
        for (int d = 0; d < k; d++)                                           \
            g_hist[d][(f >> (d * rb)) & RMASK]++;                             \
    }                                                                         \
    UT *buf = (UT *)get_scratch((size_t)n * sizeof(UT));                      \
    if (!buf) return HYDRA_ENOMEM;                                            \
    UT *src = (UT *)a, *dst = buf;                                            \
    for (int p = 0; p < k; p++) {                                             \
        const int sh = p * rb;                                                \
        uint32_t acc = 0;                                                     \
        for (uint32_t b = 0; b <= RMASK; b++) { g_off[b] = acc; acc += g_hist[p][b]; } \
        const int64_t lim = n > PF_DIST ? n - PF_DIST : 0;                    \
        int64_t i = 0;                                                        \
        if (k == 1) {                                                         \
            for (; i < lim; i++) {                                            \
                UT fp = ((UT)src[i+PF_DIST] ^ (UT)FLIP) - mn;                 \
                __builtin_prefetch(&dst[g_off[fp & RMASK]], 1, 0);            \
                UT x = src[i];                                                \
                dst[g_off[((x ^ (UT)FLIP) - mn) & RMASK]++] = x;              \
            }                                                                 \
            for (; i < n; i++) { UT x = src[i];                               \
                dst[g_off[((x ^ (UT)FLIP) - mn) & RMASK]++] = x; }            \
        } else if (p == 0) {                                                  \
            for (; i < lim; i++) {                                            \
                UT fp = ((UT)src[i+PF_DIST] ^ (UT)FLIP) - mn;                 \
                __builtin_prefetch(&dst[g_off[fp & RMASK]], 1, 0);            \
                UT f = ((UT)src[i] ^ (UT)FLIP) - mn;                          \
                dst[g_off[f & RMASK]++] = f;                                  \
            }                                                                 \
            for (; i < n; i++) { UT f = ((UT)src[i] ^ (UT)FLIP) - mn;         \
                dst[g_off[f & RMASK]++] = f; }                                \
        } else if (p == k - 1) {                                              \
            for (; i < lim; i++) {                                            \
                __builtin_prefetch(&dst[g_off[(src[i+PF_DIST] >> sh) & RMASK]], 1, 0); \
                UT f = src[i];                                                \
                dst[g_off[(f >> sh) & RMASK]++] = (f + mn) ^ (UT)FLIP;        \
            }                                                                 \
            for (; i < n; i++) { UT f = src[i];                               \
                dst[g_off[(f >> sh) & RMASK]++] = (f + mn) ^ (UT)FLIP; }      \
        } else {                                                              \
            for (; i < lim; i++) {                                            \
                __builtin_prefetch(&dst[g_off[(src[i+PF_DIST] >> sh) & RMASK]], 1, 0); \
                UT f = src[i];                                                \
                dst[g_off[(f >> sh) & RMASK]++] = f;                          \
            }                                                                 \
            for (; i < n; i++) { UT f = src[i];                               \
                dst[g_off[(f >> sh) & RMASK]++] = f; }                        \
        }                                                                     \
        UT *t = src; src = dst; dst = t;                                      \
    }                                                                         \
    if (src != (UT *)a) memcpy(a, src, (size_t)n * sizeof(T));                \
    g_last_path = "radix"; return HYDRA_OK;                                   \
}
DEFINE_HYDRA(u64, uint64_t, uint64_t, 0ull,                  64)
DEFINE_HYDRA(i64, int64_t,  uint64_t, 0x8000000000000000ull, 64)
DEFINE_HYDRA(u32, uint32_t, uint32_t, 0u,                    32)
DEFINE_HYDRA(i32, int32_t,  uint32_t, 0x80000000u,           32)
