# Profiling Notes

## Current state (April 2026)

- nsys 2024.6.2 (CUDA 12.8 toolkit) captures host-side CUDA API trace and NVTX correctly on WSL2 + 5070 Ti, but lacks GPU-side device records for Blackwell (sm_120).
- nsys 2025.6.3 (standalone) supports Blackwell GPU records but its CUPTI does not penetrate WSL2's GPU passthrough — only nsys's own bootstrap calls are recorded.
- This is a WSL2 + nsys + Blackwell incompatibility independent of our project.

## What we use locally (5070 Ti, WSL2)

- `nsys` from CUDA 12.8 toolkit for host-side API tracing + NVTX ranges.
- Wall-clock timing via `@elapsed` in Julia and `chrono` in C++.
- For correctness work and iteration speed, this is sufficient.

## What we use for headline measurements (Unity H100, future)

- nsys 2025.x with full GPU-side instrumentation against H100 SXM5 / NVL.
- This is the configuration that produces the canonical numbers in the eventual write-up.
- All benchmark-sweep results and end-to-end Sunny integration timings will be captured on Unity, not WSL2.
