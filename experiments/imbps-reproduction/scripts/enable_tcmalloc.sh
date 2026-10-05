#!/usr/bin/env bash

readonly DEFAULT_TCMALLOC_PREFIX="/data/scratch/${USER}/tools/gperftools-2.18.1"
export TCMALLOC_PREFIX="${TCMALLOC_PREFIX:-${DEFAULT_TCMALLOC_PREFIX}}"

tcmalloc_library=""
for candidate in \
  "${TCMALLOC_PREFIX}/lib/libtcmalloc.so.4" \
  "${TCMALLOC_PREFIX}/lib64/libtcmalloc.so.4"; do
  if [[ -r "${candidate}" ]]; then
    tcmalloc_library="$(readlink -f "${candidate}")"
    break
  fi
done
if [[ -z "${tcmalloc_library}" ]]; then
  echo "Pinned tcmalloc is unavailable under ${TCMALLOC_PREFIX}" >&2
  return 1 2>/dev/null || exit 1
fi
if [[ -n "${LD_PRELOAD:-}" && "${LD_PRELOAD}" != "${tcmalloc_library}" ]]; then
  echo "Refusing to mix tcmalloc with inherited LD_PRELOAD=${LD_PRELOAD}" >&2
  return 1 2>/dev/null || exit 1
fi

export LD_PRELOAD="${tcmalloc_library}"
export REQUIRE_TCMALLOC=1
unset tcmalloc_library candidate
