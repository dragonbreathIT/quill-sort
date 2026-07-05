#include "quill_core.hpp"
#include <cstdio>
#include <random>
#include <chrono>
using namespace quillcore;
using clk = std::chrono::high_resolution_clock;
template<class T, class K> double ms(std::vector<T> d, K k){
    auto t0=clk::now(); k(d.data(), d.size());
    return std::chrono::duration<double,std::milli>(clk::now()-t0).count();
}
int main(){
    std::mt19937_64 rng(1);
    printf("hardware_concurrency=%d\n", default_threads());
    for(size_t n : {5000000ull, 50000000ull}){
        std::vector<int64_t> a(n); for(auto&x:a) x=(int64_t)rng();
        double s = ms(a, [](int64_t*p,size_t m){ std::sort(p,p+m); });
        double r = ms(a, [](int64_t*p,size_t m){ serial_sort<int64_t>(p,m); });
        double pr= ms(a, [](int64_t*p,size_t m){ parallel_radix<int64_t>(p,m); });
        double ps= ms(a, [](int64_t*p,size_t m){ parallel_samplesort<int64_t>(p,m); });
        printf("int64 n=%9zu  std::sort %7.1f ms | serial_radix %7.1f | par_radix %7.1f (%.2fx) | par_sample %7.1f (%.2fx)\n",
               n, s, r, pr, s/pr, ps, s/ps);
    }
}
