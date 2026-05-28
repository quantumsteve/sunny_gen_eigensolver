module SunnyGeneigBench

using CUDA

# Path to the libsunny_geneig.so built by the C++ project. Override with
# SUNNY_GENEIG_LIBPATH if the build directory lives elsewhere.
const LIBSUNNY_GENEIG = Ref{String}("")

function __init__()
    # libsunny_geneig.so moved from build/src/ to build/src/common/.
    libpath = get(ENV, "SUNNY_GENEIG_LIBPATH",
                  joinpath(@__DIR__, "..", "..", "..", "build", "src",
                           "common", "libsunny_geneig.so"))
    LIBSUNNY_GENEIG[] = abspath(libpath)
    isfile(LIBSUNNY_GENEIG[]) ||
        error("libsunny_geneig.so not found at $(LIBSUNNY_GENEIG[]). " *
              "Build the C++ project first (cmake --build build).")
end

# C ABI status / mode constants — must match c_api.h.
const STATUS_OK            = Cint(0)
const STATUS_INVALID_ARG   = Cint(-1)
const STATUS_DEVICE_ERROR  = Cint(-2)
const STATUS_COMPILE_ERROR = Cint(-3)
const STATUS_LAUNCH_ERROR  = Cint(-4)

const TUNING_SUGGESTED = Cint(0)
const TUNING_AUTOTUNE  = Cint(1)

const CACHE_AUTO    = Cint(0)
const CACHE_NODISK  = Cint(1)
const CACHE_NOCACHE = Cint(2)

const KERNEL_UNIFIED = Cint(0)
const KERNEL_HYBRID  = Cint(1)

mutable struct Solver
    handle::Ptr{Cvoid}
    n::Int
    kernel::Symbol
    function Solver(n::Int; device_id::Int=0, bpb_request::Int=0,
                    tuning::Symbol=:suggested, cache::Symbol=:auto,
                    kernel::Symbol=:unified)
        tuning_int = (tuning === :autotune) ? TUNING_AUTOTUNE : TUNING_SUGGESTED
        cache_int  = (cache === :nodisk)    ? CACHE_NODISK    :
                     (cache === :nocache)   ? CACHE_NOCACHE   : CACHE_AUTO
        kernel_int = (kernel === :hybrid)   ? KERNEL_HYBRID   : KERNEL_UNIFIED
        h = ccall((:sunny_geneig_create_ex, LIBSUNNY_GENEIG[]),
                  Ptr{Cvoid},
                  (Cint, Cint, Cint, Cint, Cint, Cint),
                  n, device_id, bpb_request, tuning_int, cache_int, kernel_int)
        h == C_NULL && error("sunny_geneig_create_ex failed: $(last_error())")
        solver = new(h, n, kernel)
        finalizer(_destroy, solver)
        return solver
    end
end

function _destroy(s::Solver)
    h = s.handle
    if h != C_NULL
        ccall((:sunny_geneig_destroy, LIBSUNNY_GENEIG[]),
              Cvoid, (Ptr{Cvoid},), h)
        s.handle = C_NULL
    end
    return nothing
end

last_error() = unsafe_string(ccall((:sunny_geneig_last_error, LIBSUNNY_GENEIG[]),
                                    Cstring, ()))

block_dim_x(s::Solver) = Int(ccall((:sunny_geneig_block_dim_x, LIBSUNNY_GENEIG[]),
                                    Cint, (Ptr{Cvoid},), s.handle))

batches_per_block(s::Solver) = Int(ccall((:sunny_geneig_batches_per_block,
                                           LIBSUNNY_GENEIG[]),
                                          Cint, (Ptr{Cvoid},), s.handle))

shared_mem_bytes(s::Solver) = ccall((:sunny_geneig_shared_mem_bytes,
                                      LIBSUNNY_GENEIG[]),
                                     Cuint, (Ptr{Cvoid},), s.handle)

# path introspection.
path_name(s::Solver) = unsafe_string(ccall((:sunny_geneig_path_name,
                                             LIBSUNNY_GENEIG[]),
                                            Cstring, (Ptr{Cvoid},), s.handle))

function kernel_mode(s::Solver)
    v = ccall((:sunny_geneig_kernel_mode, LIBSUNNY_GENEIG[]),
              Cint, (Ptr{Cvoid},), s.handle)
    return (v == KERNEL_HYBRID) ? :hybrid : :unified
end

"""
    run_kernel!(solver, H, S; W=nothing, U=nothing, info=nothing, stream=CUDA.stream())

Run the NVRTC-compiled generalized eigenvalue solver on a batch of (H, S) pairs.
H and S are (n, n, batch) ComplexF64 CuArrays. Returns (W, U) where
W::CuMatrix{Float64} of shape (n, batch) holds ascending eigenvalues per column
and U::CuArray{ComplexF64,3} of shape (n, n, batch) holds the corresponding
generalized eigenvectors. Output buffers may be passed in to allow reuse
across timed samples (avoids re-allocation in the hot path).
"""
function run_kernel!(solver::Solver,
                     H::CuArray{ComplexF64,3},
                     S::CuArray{ComplexF64,3};
                     W::Union{CuMatrix{Float64},Nothing}=nothing,
                     U::Union{CuArray{ComplexF64,3},Nothing}=nothing,
                     info::Union{CuVector{Cint},Nothing}=nothing,
                     stream::CuStream=CUDA.stream())
    n, n2, batch = size(H)
    @assert n == n2 == solver.n  "matrix size mismatch: solver.n=$(solver.n), H=$(size(H))"
    @assert size(S) == (n, n, batch) "S size must match H"

    W_out    = W    === nothing ? CUDA.zeros(Float64,    n, batch)         : W
    U_out    = U    === nothing ? CUDA.zeros(ComplexF64, n, n, batch)      : U
    info_out = info === nothing ? CUDA.zeros(Cint,       batch)            : info

    d_H    = UInt(reinterpret(UInt, pointer(H)))
    d_S    = UInt(reinterpret(UInt, pointer(S)))
    d_W    = UInt(reinterpret(UInt, pointer(W_out)))
    d_U    = UInt(reinterpret(UInt, pointer(U_out)))
    d_info = UInt(reinterpret(UInt, pointer(info_out)))
    d_str  = UInt(reinterpret(UInt, stream.handle))

    status = ccall((:sunny_geneig_launch, LIBSUNNY_GENEIG[]),
                   Cint,
                   (Ptr{Cvoid}, UInt, UInt, UInt, UInt, UInt, Cint, UInt),
                   solver.handle, d_H, d_S, d_W, d_U, d_info,
                   batch, d_str)
    status != STATUS_OK && error("sunny_geneig_launch failed (status=$status): $(last_error())")
    return W_out, U_out, info_out
end

"""
    run_kernel!(solver, H, S, W, U, info; stream=CUDA.stream())

Positional-arg variant: caller supplies pre-allocated W (n×batch),
U (n×n×batch), info (length-batch) device buffers. Avoids any keyword-
arg / NamedTuple overhead in the hot path. Returns nothing; outputs
are written through the provided buffers.
"""
function run_kernel!(solver::Solver,
                     H::CuArray{ComplexF64,3},
                     S::CuArray{ComplexF64,3},
                     W::CuMatrix{Float64},
                     U::CuArray{ComplexF64,3},
                     info::CuVector{Cint};
                     stream::CuStream=CUDA.stream())
    n, n2, batch = size(H)
    @assert n == n2 == solver.n  "matrix size mismatch"
    @assert size(S) == (n, n, batch)        "S size must match H"
    @assert size(W) == (n, batch)            "W must be (n, batch)"
    @assert size(U) == (n, n, batch)         "U must be (n, n, batch)"
    @assert length(info) == batch            "info must be length batch"

    d_H    = UInt(reinterpret(UInt, pointer(H)))
    d_S    = UInt(reinterpret(UInt, pointer(S)))
    d_W    = UInt(reinterpret(UInt, pointer(W)))
    d_U    = UInt(reinterpret(UInt, pointer(U)))
    d_info = UInt(reinterpret(UInt, pointer(info)))
    d_str  = UInt(reinterpret(UInt, stream.handle))

    status = ccall((:sunny_geneig_launch, LIBSUNNY_GENEIG[]),
                   Cint,
                   (Ptr{Cvoid}, UInt, UInt, UInt, UInt, UInt, Cint, UInt),
                   solver.handle, d_H, d_S, d_W, d_U, d_info,
                   batch, d_str)
    status != STATUS_OK && error("sunny_geneig_launch failed (status=$status): $(last_error())")
    return nothing
end

"""
    run_kernel_chunked!(solver, H, S, W, U, info, num_streams, chunk_size)

— round-robin chunked launch over `num_streams` CUDA
streams. Splits the leading-axis-3 batch dimension into chunks of size
`chunk_size` (last chunk may be smaller), assigns chunk `k` to stream
`streams[1 + k % num_streams]`, and issues `sunny_geneig_launch` for each
chunk WITHOUT inter-chunk synchronization. Joins on all streams at the
end. Returns nothing; outputs are written through W, U, info as in the
single-launch path.

Slice semantics: each chunk uses `pointer(H, offset)` arithmetic to point
into a contiguous slab of the column-major 3-D array. No view objects
are constructed; we compute the byte-offset directly from the strided
layout. This works because the leading-axis-3 batch axis is the slowest-
varying dimension in column-major Julia, so each chunk's matrices live
in a contiguous run of memory.
"""
function run_kernel_chunked!(solver::Solver,
                              H::CuArray{ComplexF64,3},
                              S::CuArray{ComplexF64,3},
                              W::CuMatrix{Float64},
                              U::CuArray{ComplexF64,3},
                              info::CuVector{Cint},
                              num_streams::Int,
                              chunk_size::Int)
    n, n2, batch = size(H)
    @assert n == n2 == solver.n  "matrix size mismatch"
    @assert size(S) == (n, n, batch) "S size must match H"
    @assert size(W) == (n, batch)
    @assert size(U) == (n, n, batch)
    @assert length(info) == batch
    @assert num_streams >= 1
    @assert chunk_size  >= 1

    # Pre-allocate streams (CUDA.jl wraps cuStreamCreate). Stream creation
    # is cheap (~µs each) but doing it inside the timing loop would add
    # noise; the caller-side benchmark constructs streams once outside.
    streams = CuStream[CuStream() for _ in 1:num_streams]

    # Byte offsets per chunk start.
    bytes_per_mat_HSU = sizeof(ComplexF64) * n * n
    bytes_per_col_W   = sizeof(Float64)   * n
    bytes_per_info    = sizeof(Cint)

    chunk_idx = 0
    offset    = 0
    while offset < batch
        bs   = min(chunk_size, batch - offset)
        s    = streams[1 + (chunk_idx % num_streams)]

        # ccall arguments must be UInt device pointers offset into each
        # buffer. pointer(H, idx_lin) with linear index advances by element-
        # size; for a 3-D array, element idx_lin = offset*n*n + 1.
        d_H    = UInt(reinterpret(UInt, pointer(H, offset * n * n + 1)))
        d_S    = UInt(reinterpret(UInt, pointer(S, offset * n * n + 1)))
        d_W    = UInt(reinterpret(UInt, pointer(W, offset * n + 1)))
        d_U    = UInt(reinterpret(UInt, pointer(U, offset * n * n + 1)))
        d_info = UInt(reinterpret(UInt, pointer(info, offset + 1)))
        d_str  = UInt(reinterpret(UInt, s.handle))

        status = ccall((:sunny_geneig_launch, LIBSUNNY_GENEIG[]),
                       Cint,
                       (Ptr{Cvoid}, UInt, UInt, UInt, UInt, UInt, Cint, UInt),
                       solver.handle, d_H, d_S, d_W, d_U, d_info, bs, d_str)
        status != STATUS_OK &&
            error("sunny_geneig_launch (chunk $chunk_idx) failed: $(last_error())")

        chunk_idx += 1
        offset    += bs
    end

    # Join: synchronize each stream so the host barrier sees all chunks done.
    for s in streams
        CUDA.synchronize(s)
    end

    return nothing
end

export Solver, run_kernel!, run_kernel_chunked!, block_dim_x,
       batches_per_block, shared_mem_bytes, last_error,
       path_name, kernel_mode

end # module
