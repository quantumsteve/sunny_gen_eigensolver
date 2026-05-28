#!/usr/bin/env bash
# Clean Julia launcher.
#
# Forces CUDA.jl to use its JLL-managed CUDA runtime/compiler instead of any
# system-wide CUDA Toolkit on LD_LIBRARY_PATH. The system CUDA Toolkit is still
# needed by our C++/nvcc build, but for Julia we want CUDA.jl entirely
# self-contained so versioninfo() reports a single consistent runtime.
#
# Usage: ./scripts/julia_clean.sh [julia args...]

set -eu

# Make sure juliaup's julia is reachable even from non-interactive shells
# that don't source ~/.bashrc / ~/.profile.
if [ -x "$HOME/.juliaup/bin/julia" ]; then
    case ":${PATH:-}:" in
        *":$HOME/.juliaup/bin:"*) ;;
        *) export PATH="$HOME/.juliaup/bin${PATH:+:$PATH}" ;;
    esac
fi

# Strip any element of LD_LIBRARY_PATH that points into a CUDA install.
# Pattern matches /usr/local/cuda*, /opt/cuda*, /usr/lib/cuda*, plus any path
# that happens to contain "/cuda-" or "/cuda/lib".
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
    cleaned=""
    IFS=':' read -ra parts <<< "$LD_LIBRARY_PATH"
    for p in "${parts[@]}"; do
        case "$p" in
            ""|/usr/local/cuda*|/opt/cuda*|/usr/lib/cuda*|*"/cuda-"*|*"/cuda/lib"*)
                ;;
            *)
                if [ -z "$cleaned" ]; then
                    cleaned="$p"
                else
                    cleaned="$cleaned:$p"
                fi
                ;;
        esac
    done
    if [ -z "$cleaned" ]; then
        unset LD_LIBRARY_PATH
    else
        export LD_LIBRARY_PATH="$cleaned"
    fi
fi

# Also unset CUDA_PATH / CUDA_HOME so CUDA.jl's autodetection cannot re-find
# the system toolkit. With no CUDA paths visible, CUDA.jl falls back to its
# JLL artifacts — which is what we want.
unset CUDA_PATH CUDA_HOME CUDA_ROOT 2>/dev/null || true

exec julia "$@"
