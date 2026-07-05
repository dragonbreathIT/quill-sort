#!/usr/bin/env bash
# Sync the canonical companions/_core/quill_core.hpp into every package's src/.
# Edit _core/quill_core.hpp, then run this to propagate to all packages.
set -euo pipefail
cd "$(dirname "$0")"
for pkg in quill-fastsort quill-fastsort-parallel quill-fastsort-ips4o quill-fastsort-numa quill-fastsort-simd; do
  mkdir -p "$pkg/src"
  cp _core/quill_core.hpp "$pkg/src/quill_core.hpp"
done
echo "synced quill_core.hpp -> all package src/ dirs"
