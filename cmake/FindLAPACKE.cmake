# FindLAPACKE.cmake
#
# Locates LAPACKE (the C-language interface to LAPACK) and a backing
# LAPACK/BLAS implementation (OpenBLAS by default on Ubuntu).
#
# Outputs:
#   LAPACKE_FOUND
#   LAPACKE_INCLUDE_DIR
#   LAPACKE_LIBRARY
#   LAPACKE_BLAS_LIBRARY    (the BLAS/LAPACK backend, e.g. libopenblas)
#
# Imported targets:
#   LAPACK::lapacke — INTERFACE target with includes + lapacke + openblas link.

find_path(LAPACKE_INCLUDE_DIR
    NAMES lapacke.h
    HINTS /usr/include /usr/local/include
)

find_library(LAPACKE_LIBRARY
    NAMES lapacke
    HINTS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib
)

find_library(LAPACKE_BLAS_LIBRARY
    NAMES openblas
    HINTS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LAPACKE
    REQUIRED_VARS
        LAPACKE_INCLUDE_DIR
        LAPACKE_LIBRARY
        LAPACKE_BLAS_LIBRARY
)

if(LAPACKE_FOUND AND NOT TARGET LAPACK::lapacke)
    add_library(LAPACK::lapacke INTERFACE IMPORTED)
    set_target_properties(LAPACK::lapacke PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${LAPACKE_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES      "${LAPACKE_LIBRARY};${LAPACKE_BLAS_LIBRARY}"
    )
endif()

mark_as_advanced(
    LAPACKE_INCLUDE_DIR
    LAPACKE_LIBRARY
    LAPACKE_BLAS_LIBRARY
)
