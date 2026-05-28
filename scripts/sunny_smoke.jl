# Step 2 smoke test
# Verifies that the MAIQMag/Sunny.jl fork loads correctly with both
# CPU-only paths and the CUDAExt extension, on Julia 1.12.

println("Julia version: ", VERSION)
println("Loading Sunny (CPU only)...")
using Sunny
println("  OK")

# Smallest possible Sunny computation: build a tiny crystal,
# System, and energy. No GPU touched.
println("CPU sanity: building a 2-site cubic FM and computing energy...")
latvecs = lattice_vectors(1.0, 1.0, 1.0, 90, 90, 90)
crystal = Crystal(latvecs, [[0,0,0]], 1)  # P1 spacegroup
sys = System(crystal, [1 => Moment(s=1, g=2)], :dipole; dims=(2,1,1))
set_exchange!(sys, -1.0, Bond(1,1,(1,0,0)))
randomize_spins!(sys)
E = energy(sys)
println("  CPU energy = ", E, "  (finite real number expected)")

println()
println("Loading CUDA + Sunny (extension test)...")
using CUDA
ext = Base.get_extension(Sunny, :CUDAExt)
if ext === nothing
    error("CUDAExt failed to load - extension is nothing")
end
println("  CUDAExt module loaded: ", ext)

println()
println("CUDA device check:")
CUDA.versioninfo()

println()
println("All smoke checks passed.")
