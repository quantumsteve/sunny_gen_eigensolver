#!/usr/bin/env bash
# Sourceable environment script for sunny_gen_eigensolver.
# Usage: source scripts/env.sh
#
# - Sets MATHDX_ROOT to the default install path if unset.
# - Verifies cusolverdx.hpp is reachable.
# - Does NOT touch PATH or LD_LIBRARY_PATH.

set -u

if [ -z "${MATHDX_ROOT:-}" ]; then
    export MATHDX_ROOT="$HOME/opt/nvidia-mathdx-25.12.1-cuda12"
fi

echo "MATHDX_ROOT=$MATHDX_ROOT"

_cusolverdx_header="$MATHDX_ROOT/nvidia/mathdx/25.12/include/cusolverdx.hpp"
if [ ! -f "$_cusolverdx_header" ]; then
    echo "error: cusolverdx.hpp not found at $_cusolverdx_header" >&2
    return 1 2>/dev/null || exit 1
fi
unset _cusolverdx_header
export CMAKE_BUILD_PARALLEL_LEVEL=2
