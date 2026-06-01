# FindMathDx.cmake
#
# Locates the NVIDIA MathDx package (cuSolverDx + bundled CUTLASS).
#
# Inputs:
#   MathDx_ROOT       (CMake variable) — preferred
#   MATHDX_ROOT       (environment)    — fallback
#
# Outputs:
#   MathDx_FOUND
#   MathDx_VERSION
#   MathDx_INCLUDE_DIR
#   MathDx_CUTLASS_INCLUDE_DIR
#   MathDx_LIBRARY
#   MathDx_FATBIN
#
# Imported targets:
#   MathDx::cusolverdx — INTERFACE target with includes + static link

if(NOT MathDx_ROOT)
    if(DEFINED ENV{MATHDX_ROOT})
        set(MathDx_ROOT "$ENV{MATHDX_ROOT}")
    endif()
endif()

set(_mathdx_subpath "nvidia/mathdx/26.03")

find_path(MathDx_INCLUDE_DIR
    NAMES cusolverdx.hpp
    HINTS "${MathDx_ROOT}/${_mathdx_subpath}/include"
    NO_DEFAULT_PATH
)

find_path(MathDx_CUTLASS_INCLUDE_DIR
    NAMES cutlass/cutlass.h
    HINTS "${MathDx_ROOT}/${_mathdx_subpath}/external/cutlass/include"
    NO_DEFAULT_PATH
)

find_library(MathDx_LIBRARY
    NAMES cusolverdx
    HINTS "${MathDx_ROOT}/${_mathdx_subpath}/lib"
    NO_DEFAULT_PATH
)

find_file(MathDx_FATBIN
    NAMES libcusolverdx.fatbin
    HINTS "${MathDx_ROOT}/${_mathdx_subpath}/lib"
    NO_DEFAULT_PATH
)

if(MathDx_INCLUDE_DIR AND MathDx_CUTLASS_INCLUDE_DIR AND MathDx_LIBRARY AND MathDx_FATBIN)
    set(MathDx_VERSION "26.03.0")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MathDx
    REQUIRED_VARS
        MathDx_INCLUDE_DIR
        MathDx_CUTLASS_INCLUDE_DIR
        MathDx_LIBRARY
        MathDx_FATBIN
    VERSION_VAR MathDx_VERSION
)

if(MathDx_FOUND AND NOT TARGET MathDx::cusolverdx)
    add_library(MathDx::cusolverdx INTERFACE IMPORTED)
    set_target_properties(MathDx::cusolverdx PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES
            "${MathDx_INCLUDE_DIR};${MathDx_CUTLASS_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES
            "${MathDx_LIBRARY}"
    )
endif()

# MathDx_FATBIN is intentionally not advanced — NVRTC consumers need
# its absolute path injected as a compile definition.
mark_as_advanced(
    MathDx_INCLUDE_DIR
    MathDx_CUTLASS_INCLUDE_DIR
    MathDx_LIBRARY
)
