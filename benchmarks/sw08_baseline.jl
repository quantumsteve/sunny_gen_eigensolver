# Baseline reproduction of MAIQMag/Sunny.jl SW08_sqrt3_kagome_AFM_CUDA.
# Source: /home/rb/projects/Sunny.jl/examples/spinw_tutorials/SW08_sqrt3_kagome_AFM_CUDA.jl
# Step 3: establishing the cuSolverDx project's reference baseline.
# Modifications:
#   - Plotting calls commented out (headless WSL)
#   - Wall-clock @time wrapped around the hot calls (already present upstream;
#     left intact)
#   - NVTX ranges added (see scheme below)
#   - End-of-script summary block prints grep-friendly timing lines
# Original line numbers for the modified calls preserved as comments.
#
# NVTX scheme:
#   - Outer range "powder_average_b{N}" wraps each ext.powder_average(...) call.
#   - Inner range "geneig_batch" wraps the intensities(swt_d, qs; ...) call
#     inside the do-block; this is the per-batch path that ultimately calls
#     hegvd_batched! in ext/CUDAExt/EigenBatched.jl, the routine our
#     cuSolverDx in-kernel solve will eventually replace.

# # SW08 - √3×√3 kagome antiferromagnet
#
# This is a Sunny port of [SpinW Tutorial
# 8](https://spinw.org/tutorials/08tutorial), originally authored by Bjorn Fak
# and Sandor Toth. It calculates the linear spin wave theory spectrum for the
# ``\sqrt{3} \times \sqrt{3}`` order of a kagome antiferromagnet.

# Load Sunny and the GLMakie plotting package.

using CUDA
#CUDA.set_runtime_version!(v"12.9"; local_toolkit=true)
using Sunny # commented out for headless run: , GLMakie
using NVTX
using Printf
using Dates
using JSON3

# Define the chemical cell of a kagome lattice with spacegroup 147 (P-3).

units = Units(:meV, :angstrom)
latvecs = lattice_vectors(6, 6, 40, 90, 90, 120)
cryst = Crystal(latvecs, [[1/2, 0, 0]], 147)
#view_crystal(cryst; ndims=2)

# Construct a spin system with nearest neighbor antiferromagnetic exchange.

sys = System(cryst, [1 => Moment(s=1, g=2)], :dipole)
J = 1.0
set_exchange!(sys, J, Bond(2, 3, [0, 0, 0]))

# Initialize to an energy minimizing magnetic structure, for which
# nearest-neighbor spins are at 120° angles.

set_dipole!(sys, [cos(0), sin(0), 0], (1, 1, 1, 1))
set_dipole!(sys, [cos(0), sin(0), 0], (1, 1, 1, 2))
set_dipole!(sys, [cos(2π/3), sin(2π/3), 0], (1, 1, 1, 3))
k = [-1/3, -1/3, 0]
axis = [0, 0, 1]
sys_enlarged = repeat_periodically_as_spiral(sys, (3, 3, 1); k, axis)
#plot_spins(sys_enlarged; ndims=2)

# Check energy per site. Each site participates in 4 bonds with energy
# ``J\cos(2π/3)``. Factor of 1/2 avoids double counting. The two calculation
# methods agree.

@assert energy_per_site(sys_enlarged) ≈ (4/2)*J*cos(2π/3)
@assert spiral_energy_per_site(sys; k, axis) ≈ (4/2)*J*cos(2π/3)

# Calculate and plot intensities for a path through ``𝐪``-space using two
# calculation methods. The two methods agree in intensity, but the "supercell
# method" gives rise to ghost modes in the dispersion that have zero intensity.

qs = [[-1/2,0,0], [0,0,0], [1/2,1/2,0]]
path = q_space_path(cryst, qs, 400)

#fig = Figure(size=(768, 300))
swt = SpinWaveTheory(sys_enlarged; measure=ssf_perp(sys_enlarged))
swt_d = to_device(swt)
#res = intensities_bands(swt, path)
#res = Sunny.Intensities(res)
#plot_intensities!(fig[1, 1], res; units, saturation=0.5,title="Supercell method")
#swt = SpinWaveTheorySpiral(sys; measure=ssf_perp(sys), k, axis)
#res = intensities_bands(swt, path)
#plot_intensities!(fig[1, 2], res; units, saturation=0.5, title="Spiral method")
#fig

# Calculate and plot the powder averaged spectrum. Continuing to use the "spiral
# method", this calculation executes in about two seconds. Because the
# intensities are dominated by a flat band at zero energy transfer, select an
# empirical `colorrange` that brings the lower-intensity features into focus.

radii = range(0, 2.5, 200)
energies = range(0, 2.5, 200)
kernel = gaussian(fwhm=0.05)
kernel_d = to_device(kernel)
ext = Base.get_extension(Sunny, :CUDAExt)

# Initial powder_average call (was upstream line 73-75; batch_size=3, also
# acts as warm-up so the loop below sees fully compiled code paths).
t_initial = NVTX.@range "powder_average_initial" begin
    @elapsed global res_d = ext.powder_average(cryst, radii, 200, batch_size=3) do qs
        NVTX.@range "geneig_batch" begin
            intensities(swt_d, qs; kernel=kernel_d, energies=energies)
        end
    end
end
@printf "  powder_average_initial (batch_size=3) wall=%.4f s\n" t_initial

# Sweep batch sizes 1..3 (was upstream lines 76-81). The whole sweep is wrapped
# in "batch_size_sweep" so the trace clearly delineates the parameter sweep over
# batch_size from the warmup and the larger 360k run.
batch_times = Dict{Int, Float64}()
NVTX.@range "batch_size_sweep" begin
    for i in 1:3
        println(i)
        t = NVTX.@range "powder_average_b$i" begin
            @elapsed global res_d = ext.powder_average(cryst, radii, 200, batch_size=i) do qs
                NVTX.@range "geneig_batch" begin
                    intensities(swt_d, qs; kernel=kernel_d, energies=energies)
                end
            end
        end
        batch_times[i] = t
        @printf "  powder_average_loop iter batch_size=%d wall=%.4f s\n" i t
    end
end
res = Sunny.PowderIntensities(res_d, cryst)
# commented out for headless run: plot_intensities(res; units, colorrange=(0, 20))

# --- End-of-script timing summary (grep-friendly) ---------------------------
println()
println("==== SW08 BASELINE TIMING SUMMARY ====")
println("BASELINE radii=200 samples=200 energies=200 batched_solves_per_call=40000")
@printf "BASELINE powder_average_initial batch_size=3 wall=%.4f s\n" t_initial
for i in 1:3
    @printf "BASELINE powder_average_loop  batch_size=%d wall=%.4f s\n" i batch_times[i]
end
println("======================================")

# Steven's email-thread configuration: 360,000 q-points in batches of 12,000
# (radii = 600, samples = 600 → 360,000). Guarded by a pre-flight memory check
# so we skip cleanly on memory-constrained devices instead of OOM-crashing.
const STEVEN_MEMORY_FLOOR_GIB = 4
avail_gib = CUDA.available_memory() / (1024^3)
t_steven = nothing
res_steven = nothing
if avail_gib < STEVEN_MEMORY_FLOOR_GIB
    println("WARNING: only $(round(avail_gib, digits=2)) GiB free GPU memory; skipping 360k variant (needs ≥$(STEVEN_MEMORY_FLOOR_GIB) GiB)")
else
    println()
    println("=== 360k configuration (Steven's NVIDIA-thread reference) ===")
    radii_steven = range(0, 2.5, 600)
    t_steven = @elapsed begin
        NVTX.@range "powder_average_steven_360k" begin
            global res_steven = ext.powder_average(cryst, radii_steven, 600, batch_size=3) do qs
                NVTX.@range "geneig_batch" begin
                    intensities(swt_d, qs; kernel=kernel_d, energies=energies)
                end
            end
        end
    end
    println("BASELINE_steven_360k_seconds: ", t_steven)
end

# Machine-readable summary for later parsing/comparison.
# n_atoms_magnetic_cell uses sys_enlarged (the 27-site spiral-enlarged cell),
# not the 3-site chemical sys; that is what the eigenvalue solver actually
# operates on, so its size determines matrix_size = 2 * N.
summary = Dict(
    "host" => Sys.MACHINE,
    "julia_version" => string(VERSION),
    "cuda_runtime" => string(CUDA.runtime_version()),
    "device_name" => CUDA.name(CUDA.device()),
    "device_capability" => string(CUDA.capability(CUDA.device())),
    "n_atoms_magnetic_cell" => length(eachsite(sys_enlarged)),
    "matrix_size" => 2 * length(eachsite(sys_enlarged)),
    "t_initial_warmup_s" => t_initial,
    "t_batch_b1_s" => batch_times[1],
    "t_batch_b2_s" => batch_times[2],
    "t_batch_b3_s" => batch_times[3],
    "t_steven_360k_s" => t_steven,
    "n_qpoints_40k_sweep" => 40_000,
    "n_qpoints_steven" => 360_000,
)
open(joinpath(@__DIR__, "results", "sw08_baseline_$(Dates.format(now(), "yyyymmdd_HHMMSS")).json"), "w") do io
    JSON3.pretty(io, summary)
end
println("Results written to benchmarks/results/")
