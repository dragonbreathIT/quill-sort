// Exhaustive correctness harness for quill_core.hpp — compares every kernel
// against std::sort over many dtypes, sizes, and adversarial distributions.
#include "quill_core.hpp"
#include <cstdio>
#include <random>
#include <limits>
#include <cmath>
using namespace quillcore;

static int g_fail = 0;

template <class T>
bool eq_vec(const std::vector<T>& x, const std::vector<T>& y) {
    if (x.size() != y.size()) return false;
    for (size_t i = 0; i < x.size(); ++i) {
        if (x[i] != y[i]) {
            // treat +0/-0 and equal NaNs as matches (== semantics like numpy)
            if (std::is_floating_point<T>::value) {
                if (x[i] == y[i]) continue;
                if (std::isnan((double)x[i]) && std::isnan((double)y[i])) continue;
            }
            return false;
        }
    }
    return true;
}

template <class T, class Kernel>
void check(const char* name, const char* dist, std::vector<T> data, Kernel k) {
    std::vector<T> expect = data;
    std::sort(expect.begin(), expect.end());
    std::vector<T> got = data;
    k(got.data(), got.size());
    if (!eq_vec(expect, got)) {
        g_fail++;
        printf("  FAIL  %-14s %-12s n=%zu\n", name, dist, data.size());
    }
}

template <class T>
std::vector<T> gen(const char* dist, size_t n, std::mt19937_64& rng) {
    std::vector<T> v(n);
    auto rnd = [&]() -> T {
        if (std::is_floating_point<T>::value) {
            uint64_t bits = rng();
            T f; if (sizeof(T)==4){ uint32_t b=(uint32_t)bits; std::memcpy(&f,&b,4);} else std::memcpy(&f,&bits,8);
            if (std::isnan((double)f)) f = (T)((int64_t)bits); // avoid NaN (dispatcher strips them)
            return f;
        } else {
            return (T)rng();
        }
    };
    std::string d = dist;
    for (size_t i = 0; i < n; ++i) {
        if (d == "random")        v[i] = rnd();
        else if (d == "equal")    v[i] = (T)42;
        else if (d == "sorted")   v[i] = (T)i;
        else if (d == "reverse")  v[i] = (T)(n - i);
        else if (d == "fewunique")v[i] = (T)(rng() % 5);
        else if (d == "extremes") v[i] = (i & 1) ? std::numeric_limits<T>::max()
                                                 : std::numeric_limits<T>::lowest();
    }
    if (d == "floatspecial" && std::is_floating_point<T>::value) {
        T specials[] = { (T)0.0, (T)-0.0, std::numeric_limits<T>::infinity(),
                         -std::numeric_limits<T>::infinity(),
                         std::numeric_limits<T>::min(), std::numeric_limits<T>::lowest(),
                         (T)1.5, (T)-1.5, std::numeric_limits<T>::denorm_min() };
        for (size_t i = 0; i < n; ++i) v[i] = specials[rng() % 9];
    }
    return v;
}

template <class T>
void run(const char* name) {
    std::mt19937_64 rng(0xC0FFEE ^ std::hash<std::string>{}(name));
    const char* dists[] = {"random","equal","sorted","reverse","fewunique","extremes"};
    std::vector<size_t> sizes = {0,1,2,3,5,17,63,255,256,257,1000,65535,65536,100000,1000003};
    for (auto dist : dists) {
        for (size_t n : sizes) {
            auto base = gen<T>(dist, n, rng);
            check<T>(name, dist, base, [](T* p, size_t m){ serial_sort<T>(p, m); });
            check<T>(name, dist, base, [](T* p, size_t m){ parallel_radix<T>(p, m); });
            check<T>(name, dist, base, [](T* p, size_t m){ parallel_samplesort<T>(p, m); });
        }
    }
    if (std::is_floating_point<T>::value) {
        for (size_t n : {17u, 1000u, 100000u}) {
            auto base = gen<T>("floatspecial", n, rng);
            check<T>(name, "floatspecial", base, [](T* p, size_t m){ serial_sort<T>(p, m); });
            check<T>(name, "floatspecial", base, [](T* p, size_t m){ parallel_radix<T>(p, m); });
            check<T>(name, "floatspecial", base, [](T* p, size_t m){ parallel_samplesort<T>(p, m); });
        }
    }
    printf("  %-8s done\n", name);
}

int main() {
    printf("quill_core correctness harness\n");
    run<int32_t>("int32");
    run<uint32_t>("uint32");
    run<int64_t>("int64");
    run<uint64_t>("uint64");
    run<float>("float32");
    run<double>("float64");
    printf(g_fail ? "\nFAILURES: %d\n" : "\nALL PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
