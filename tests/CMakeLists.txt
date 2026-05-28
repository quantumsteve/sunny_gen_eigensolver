# tests organized by which path they exercise:
#   static/   — N=54-hardcoded reference kernels (links to geneig_fixed)
#   common/   — path-agnostic infrastructure: cuSolverDx repros, C ABI test,
#               heev-fraction diagnostic, XsyevBatched standalone bench
#   unified/  — UnifiedGeneigSolver tests (links to unified_geneig)
#   hybrid/   — HybridGeneigSolver tests (links to hybrid_geneig)

add_subdirectory(static)
add_subdirectory(common)
add_subdirectory(unified)
add_subdirectory(hybrid)
