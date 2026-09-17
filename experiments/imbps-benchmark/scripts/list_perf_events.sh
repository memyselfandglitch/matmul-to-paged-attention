#!/usr/bin/env bash
set -euo pipefail

if ! command -v perf >/dev/null 2>&1; then
    echo "perf is not installed" >&2
    exit 1
fi

echo "Candidate cache, memory, IBS, and data-fabric events exposed by this kernel:"
perf list 2>/dev/null | grep -Ei '(^|[[:space:]])(amd_|ibs|l3|llc|l2|dram|data.fabric|df_)' || true

echo
echo "Generic permission smoke test:"
perf stat -e cycles,instructions,cache-references,cache-misses -- true

