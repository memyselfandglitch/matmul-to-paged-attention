#!/usr/bin/env bash
set -euo pipefail

readonly GPERFTOOLS_VERSION="2.18.1"
readonly GPERFTOOLS_SHA256="d18d919175f9e4d740ace6b52f0f4f91284160c454e91b36ffd6456282a02206"
readonly ARCHIVE="gperftools-${GPERFTOOLS_VERSION}.tar.gz"
readonly DOWNLOAD_URL="https://github.com/gperftools/gperftools/releases/download/gperftools-${GPERFTOOLS_VERSION}/${ARCHIVE}"
readonly INSTALL_PREFIX="${TCMALLOC_PREFIX:-/data/scratch/${USER}/tools/gperftools-${GPERFTOOLS_VERSION}}"
readonly BUILD_JOBS="${TCMALLOC_BUILD_JOBS:-${SLURM_CPUS_PER_TASK:-8}}"

for tool in tar sha256sum make gcc g++; do
  if ! command -v "${tool}" >/dev/null 2>&1; then
    echo "Required build tool is unavailable: ${tool}" >&2
    exit 1
  fi
done
if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
  echo "curl or wget is required to download gperftools" >&2
  exit 1
fi

existing_library=""
for candidate in \
  "${INSTALL_PREFIX}/lib/libtcmalloc.so.4" \
  "${INSTALL_PREFIX}/lib64/libtcmalloc.so.4"; do
  if [[ -r "${candidate}" ]]; then
    existing_library="${candidate}"
    break
  fi
done
if [[ -n "${existing_library}" ]]; then
  echo "Pinned tcmalloc is already installed: ${existing_library}"
  exit 0
fi
if [[ -e "${INSTALL_PREFIX}" ]]; then
  echo "Refusing to overwrite incomplete installation: ${INSTALL_PREFIX}" >&2
  exit 1
fi

readonly BUILD_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/imbps-tcmalloc.XXXXXX")"
trap 'rm -rf -- "${BUILD_ROOT}"' EXIT
readonly ARCHIVE_PATH="${BUILD_ROOT}/${ARCHIVE}"

if command -v curl >/dev/null 2>&1; then
  curl --fail --location --retry 3 --output "${ARCHIVE_PATH}" "${DOWNLOAD_URL}"
else
  wget --tries=3 --output-document="${ARCHIVE_PATH}" "${DOWNLOAD_URL}"
fi
printf '%s  %s\n' "${GPERFTOOLS_SHA256}" "${ARCHIVE_PATH}" | sha256sum --check --strict

tar -xzf "${ARCHIVE_PATH}" -C "${BUILD_ROOT}"
cd "${BUILD_ROOT}/gperftools-${GPERFTOOLS_VERSION}"
./configure \
  --prefix="${INSTALL_PREFIX}" \
  --disable-static \
  --enable-shared \
  --disable-dependency-tracking
make -j "${BUILD_JOBS}"
make install

installed_library=""
for candidate in \
  "${INSTALL_PREFIX}/lib/libtcmalloc.so.4" \
  "${INSTALL_PREFIX}/lib64/libtcmalloc.so.4"; do
  if [[ -r "${candidate}" ]]; then
    installed_library="${candidate}"
    break
  fi
done
if [[ -z "${installed_library}" ]]; then
  echo "Build completed but libtcmalloc.so.4 was not installed" >&2
  exit 1
fi

echo "tcmalloc ${GPERFTOOLS_VERSION}: ${installed_library}"
sha256sum "$(readlink -f "${installed_library}")"
