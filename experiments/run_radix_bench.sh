#!/usr/bin/env bash
# Compile + run the radix scatter benchmark. Usage:
#   bash run_radix_bench.sh              # default sweep: 5M, 20M, 50M
#   bash run_radix_bench.sh 100          # 100 million elements
#   bash run_radix_bench.sh 10 50 200    # custom sizes (in millions)
set -e
cd "$(dirname "$0")"
CXX="${CXX:-c++}"                      # c++ = your system compiler (clang++/g++)
FLAGS="-std=c++17 -O3 -pthread"
# -march=native squeezes out std::sort's SIMD; drop it if your compiler rejects it.
if echo 'int main(){}' | "$CXX" $FLAGS -march=native -x c++ - -o /dev/null 2>/dev/null; then
  FLAGS="$FLAGS -march=native"
fi
echo "compiling:  $CXX $FLAGS radix_bench.cpp -o radix_bench"
"$CXX" $FLAGS radix_bench.cpp -o radix_bench
echo
./radix_bench "$@"
