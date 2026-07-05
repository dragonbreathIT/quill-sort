// radix_bench.cpp — A/B the "new idea" (write-combining radix) against quill's
// current radix and std::sort, at real C++ speed. Single-threaded on purpose:
// this isolates the SCATTER behaviour (the thing write-combining changes) from
// the parallel MSD partition. If write-combining wins here, it wins in the
// parallel kernel too.
//
// Sorts uint64 (the sign/float order-preserving transform is orthogonal — it's a
// cheap pre/post pass and doesn't affect which scatter strategy is fastest).
//
// Build + run:  bash run_radix_bench.sh           (default sweep)
//               bash run_radix_bench.sh 100        (100 million elements)
//               bash run_radix_bench.sh 10 50 200  (custom sizes, in millions)
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

using u64 = uint64_t;
using clk = std::chrono::high_resolution_clock;
static double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

// ── baseline: std::sort (introsort — the "np.sort" equivalent) ───────────────
static void bench_stdsort(u64* a, size_t n, u64*) { std::sort(a, a + n); }

// ── CURRENT approach: plain 8-bit LSD radix (what quill_core.hpp does) ────────
static void radix_naive(u64* a, size_t n, u64* tmp) {
    u64* src = a; u64* dst = tmp;
    for (int p = 0; p < 8; ++p) {
        const int sh = p * 8;
        size_t cnt[256] = {0};
        for (size_t i = 0; i < n; ++i) cnt[(src[i] >> sh) & 0xFF]++;
        size_t off[256], s = 0;
        for (int b = 0; b < 256; ++b) { off[b] = s; s += cnt[b]; }
        for (size_t i = 0; i < n; ++i)                    // <-- random scatter into dst
            dst[off[(src[i] >> sh) & 0xFF]++] = src[i];
        std::swap(src, dst);
    }
    if (src != a) std::memcpy(a, src, n * sizeof(u64));
}

// ── lever 1: wider digits -> fewer passes (11-bit -> 6 passes vs 8) ──────────
static void radix_11(u64* a, size_t n, u64* tmp) {
    constexpr int BITS = 11, R = 1 << BITS, MASK = R - 1, PASSES = (64 + BITS - 1) / BITS;
    std::vector<size_t> cnt(R), off(R);
    u64* src = a; u64* dst = tmp;
    for (int p = 0; p < PASSES; ++p) {
        const int sh = p * BITS;
        std::fill(cnt.begin(), cnt.end(), 0);
        for (size_t i = 0; i < n; ++i) cnt[(src[i] >> sh) & MASK]++;
        size_t s = 0;
        for (int b = 0; b < R; ++b) { off[b] = s; s += cnt[b]; }
        for (size_t i = 0; i < n; ++i) dst[off[(src[i] >> sh) & MASK]++] = src[i];
        std::swap(src, dst);
    }
    if ((PASSES & 1) && src != a) std::memcpy(a, src, n * sizeof(u64));
    else if (src != a) std::memcpy(a, src, n * sizeof(u64));
}

// ── the NEW idea: write-combining 8-bit radix ────────────────────────────────
// Instead of scattering single elements to 256 random locations (cache/TLB
// thrash), append into 256 small per-bucket buffers; when a buffer fills a
// cache-line-multiple, flush it to dst as one sequential burst. Random writes
// become streaming writes; the buffers stay hot in L1/L2.
template <int B>
static void radix_wc(u64* a, size_t n, u64* tmp) {
    u64* src = a; u64* dst = tmp;
    alignas(64) u64 buf[256 * B];   // per-bucket write buffers (256*B*8 bytes, stays hot in L1/L2)
    int bc[256];
    for (int p = 0; p < 8; ++p) {
        const int sh = p * 8;
        size_t cnt[256] = {0};
        for (size_t i = 0; i < n; ++i) cnt[(src[i] >> sh) & 0xFF]++;
        size_t off[256], s = 0;
        for (int b = 0; b < 256; ++b) { off[b] = s; s += cnt[b]; }
        std::memset(bc, 0, sizeof(bc));
        for (size_t i = 0; i < n; ++i) {
            const int d = (src[i] >> sh) & 0xFF;
            buf[d * B + bc[d]++] = src[i];
            if (bc[d] == B) {                             // flush a full block sequentially
                std::memcpy(dst + off[d], buf + d * B, B * sizeof(u64));
                off[d] += B; bc[d] = 0;
            }
        }
        for (int b = 0; b < 256; ++b)                     // flush partial blocks
            if (bc[b]) { std::memcpy(dst + off[b], buf + b * B, bc[b] * sizeof(u64)); off[b] += bc[b]; }
        std::swap(src, dst);
    }
    if (src != a) std::memcpy(a, src, n * sizeof(u64));
}

int main(int argc, char** argv) {
    std::vector<size_t> sizes;
    for (int i = 1; i < argc; ++i) sizes.push_back(std::stoull(argv[i]) * 1'000'000ull);
    if (sizes.empty()) sizes = {5'000'000, 20'000'000, 50'000'000};

    std::mt19937_64 rng(12345);
    printf("radix_bench — uint64, single-threaded, best of 3\n");
    printf("(speedups shown vs std::sort; last column = write-combining vs current radix)\n\n");
    printf("%12s %11s %11s %11s %11s %11s %11s   %s\n",
           "n", "std::sort", "radix_naive", "radix_11", "wc<8>", "wc<16>", "wc<32>", "wc32/naive");
    printf("%s\n", std::string(110, '-').c_str());

    for (size_t n : sizes) {
        std::vector<u64> master(n);
        for (auto& x : master) x = rng();
        std::vector<u64> ref = master; std::sort(ref.begin(), ref.end());
        std::vector<u64> tmp(n);

        auto run = [&](auto fn) {
            double best = 1e18;
            for (int r = 0; r < 3; ++r) {
                std::vector<u64> a = master;
                auto t0 = clk::now();
                fn(a.data(), n, tmp.data());
                best = std::min(best, ms_since(t0));
                if (a != ref) { printf("  !! INCORRECT result\n"); break; }
            }
            return best;
        };

        double t_std = run(bench_stdsort);
        double t_nai = run(radix_naive);
        double t_11  = run(radix_11);
        double t_w8  = run(radix_wc<8>);
        double t_w16 = run(radix_wc<16>);
        double t_w32 = run(radix_wc<32>);

        auto sp = [&](double t){ return t_std / t; };
        printf("%12zu %9.1fms %8.1f(%.1fx) %6.1f(%.1fx) %5.1f(%.1fx) %5.1f(%.1fx) %5.1f(%.1fx)   %.2fx\n",
               n, t_std, t_nai, sp(t_nai), t_11, sp(t_11),
               t_w8, sp(t_w8), t_w16, sp(t_w16), t_w32, sp(t_w32), t_nai / t_w32);
    }
    printf("\nRead: >1.0x beats std::sort. Final column >1.0 means write-combining\n"
           "beats the current plain radix — that's the 'new thing' earning its keep.\n");
    return 0;
}
