# SW08 sqrt(3)×sqrt(3) kagome AFM setup, mirroring
# /home/rb/projects/Sunny.jl/examples/spinw_tutorials/SW08_sqrt3_kagome_AFM_CUDA.jl
# up to the SpinWaveTheoryDevice. Provides build_sw08_matrices(swt_d, Nq) which
# returns the (H_d, Σ_d) pair PR#8's hegvd_batched! consumes:
#
#   H_d : (2L, 2L, Nq) ComplexF64 CuArray — dynamical matrix populated by
#         Sunny.dynamical_matrix!. This is the positive-definite operator
#         that PR#8's pipeline Cholesky-factors.
#   Σ_d : (2L, 2L, Nq) ComplexF64 CuArray — the **paraunitary metric**
#         diag(I_L, -I_L) replicated across the batch axis. Sunny initializes
#         this via `_set_identity` (see ext/CUDAExt/SpinWaveTheory/DispersionAndIntensities.jl:3),
#         which actually places +1 on the first L diagonal entries and -1 on
#         the next L. Calling it "identity" is a misnomer; it is the Colpa
#         method's Σ.
#
# For our project's NVRTC kernel which solves H u = λ S u, the equivalent
# call is solver.launch(H = Σ_d, S = H_d) — gives the same eigenvalue spectrum
# as PR#8 because both are computing eigendecompositions of L⁻¹ Σ L⁻*.

using CUDA
using Sunny
using LinearAlgebra

"""
    build_sw08_swt_device()

Replicate the SW08 SpinWaveTheoryDevice exactly. Returns `(swt_d, sys, cryst)`.
"""
function build_sw08_swt_device()
    units = Units(:meV, :angstrom)
    latvecs = lattice_vectors(6, 6, 40, 90, 90, 120)
    cryst = Crystal(latvecs, [[1/2, 0, 0]], 147)

    sys = System(cryst, [1 => Moment(s=1, g=2)], :dipole)
    J = 1.0
    set_exchange!(sys, J, Bond(2, 3, [0, 0, 0]))

    set_dipole!(sys, [cos(0), sin(0), 0], (1, 1, 1, 1))
    set_dipole!(sys, [cos(0), sin(0), 0], (1, 1, 1, 2))
    set_dipole!(sys, [cos(2π/3), sin(2π/3), 0], (1, 1, 1, 3))
    k = [-1/3, -1/3, 0]
    axis = [0, 0, 1]
    sys_enlarged = repeat_periodically_as_spiral(sys, (3, 3, 1); k, axis)

    swt = SpinWaveTheory(sys_enlarged; measure=ssf_perp(sys_enlarged))
    swt_d = to_device(swt)
    return (swt_d, swt, sys_enlarged, cryst)
end

# CUDA kernel that paints Σ = diag(I_L, -I_L) onto each (n,n) slab of a
# (n, n, Nq) array. Identical to Sunny's CUDAExt _set_identity; copied here
# rather than imported because Sunny doesn't export it (it's private to
# the CUDAExt module).
function _set_metric_kernel(a)
    i = threadIdx().x + (blockIdx().x - Int32(1)) * blockDim().x
    if i > size(a, 2); return; end
    iq = threadIdx().y + (blockIdx().y - Int32(1)) * blockDim().y
    if iq > size(a, 3); return; end
    L = div(size(a, 1), 2)
    a[i, i, iq] = i > L ? -1. : 1.
    return
end

function set_paraunitary_metric!(M::CuArray{ComplexF64,3})
    twoL, _, Nq = size(M)
    fill!(M, 0)
    kernel = @cuda launch=false _set_metric_kernel(M)
    config = launch_configuration(kernel.fun)
    optimal_threads_1d = config.threads
    tx = min(twoL, optimal_threads_1d)
    ty = min(Nq, optimal_threads_1d ÷ tx)
    bx = cld(twoL, tx)
    by = cld(Nq, ty)
    kernel(M; threads=(tx, ty), blocks=(bx, by))
    return M
end

"""
    build_sw08_matrices(swt_d, swt_host, sys_enlarged, cryst, Nq; rng=...)

Generate Nq random q-points (in the original crystal's reciprocal lattice
units, uniform on [-0.5, 0.5]^3 — same domain Sunny uses for path/powder
calculations) and populate (H_d, Σ_d) via Sunny.dynamical_matrix!. The
output pair is exactly what PR#8's hegvd_batched! consumes.
"""
function build_sw08_matrices(swt_d, swt_host, sys_enlarged, cryst, Nq;
                              seed::UInt64=0xC0FFEE0042 % UInt64)
    L = Sunny.nbands(swt_d)
    twoL = 2 * L

    # Random q-points in original-crystal RLU.
    rng = Random.MersenneTwister(seed)
    qs_cpu = [Sunny.Vec3(rand(rng, 3) .- 0.5) for _ in 1:Nq]
    # Sunny's CUDAExt path stores qpts.qs directly as a CuVector for
    # the dynamical_matrix! kernel. We replicate that.
    qs_d = CuArray(qs_cpu)

    # Same reshaped_rlu transform Sunny applies internally.
    sys = sys_enlarged
    reshaped_rlu = inv(2π) * sys.crystal.latvecs' * cryst.recipvecs

    H_d = CUDA.zeros(ComplexF64, twoL, twoL, Nq)
    Sunny.dynamical_matrix!(H_d, swt_d, reshaped_rlu, qs_d)

    Σ_d = CUDA.zeros(ComplexF64, twoL, twoL, Nq)
    set_paraunitary_metric!(Σ_d)

    return (H_d, Σ_d)
end

# Random.jl is not in our Project.toml; pull it via stdlib.
using Random
