project(miniVite)
cmake_minimum_required(VERSION 3.10)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
    message(STATUS "CMAKE_BUILD_TYPE is set as Release")
endif ()

# ---------- CPP ---------- #
if (RUN_BUILD_AND_TEST_WITH_CI)
    set(CMAKE_CXX_FLAGS "-w -pthread")
else ()
    set(CMAKE_CXX_FLAGS "-Wall -pthread")
endif ()

# Debug
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3 -Wextra")
if (Linux)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -pg")
endif ()

# Release
if (RUN_BUILD_AND_TEST_WITH_CI)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
else ()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
endif ()

# Release with debug info
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g3 -Og")
if (Linux)
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -pg")
endif ()

# ---------- C ---------- #
if (RUN_BUILD_AND_TEST_WITH_CI)
    set(CMAKE_C_FLAGS "-w -pthread")
else ()
    set(CMAKE_C_FLAGS "-Wall -pthread")
endif ()

# Debug
set(CMAKE_C_FLAGS_DEBUG "-O0 -g3 -Wextra")
if (Linux)
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -pg")
endif ()

# Release
if (RUN_BUILD_AND_TEST_WITH_CI)
    set(CMAKE_C_FLAGS_RELEASE "-O3")
else ()
    set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
endif ()

# Release with debug info
set(CMAKE_C_FLAGS_RELWITHDEBINFO "-g3 -Og")
if (Linux)
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -pg")
endif ()

# ---------- Boost ---------- #
find_package(Boost 1.64)

# ---------- Metall ---------- #
if (USE_METALL)
    if (METALL_ROOT)
        find_path(METALL_INCLUDE_DIR metall/metall.hpp PATHS ${METALL_ROOT}/include REQUIRED)
    else ()
        find_path(METALL_INCLUDE_DIR metall/metall.hpp REQUIRED)
    endif ()

    if (${METALL_INCLUDE_DIR} STREQUAL "METALL_INCLUDE_DIR-NOTFOUND")
        message(FATAL_ERROR "Cannot find Metall")
    endif()

    if (NOT Boost_FOUND)
        message(FATAL_ERROR "Cannot find Boost C++ Libraries --- Metall requires them")
    endif()

    # C++ filesystem #
    include(include_cxx_filesystem_library)
    include_cxx_filesystem_library()
endif ()

macro(with_metall target)
    target_include_directories(${target} PUBLIC ${Boost_INCLUDE_DIRS})
    target_include_directories(${target} PUBLIC ${METALL_INCLUDE_DIR})
    target_compile_definitions(${target} PUBLIC USE_METALL_DSTORE)
endmacro()

# ---------- Umap ---------- #
if (USE_UMAP)
    if (UMAP_ROOT)
        find_library(LIBUMAP umap PATHS ${UMAP_ROOT}/lib REQUIRED)
        find_path(UMAP_INCLUDE_DIR umap/umap.h PATHS ${UMAP_ROOT}/include REQUIRED)
    else ()
        find_library(LIBUMAP umap REQUIRED)
        find_path(UMAP_INCLUDE_DIR umap/umap.h REQUIRED)
    endif ()

    if ((${LIBUMAP} STREQUAL "LIBUMAP-NOTFOUND") OR (${UMAP_INCLUDE_DIR} STREQUAL "UMAP_INCLUDE_DIR-NOTFOUND"))
        message(FATAL_ERROR "Cannot find Umap")
    endif()
endif ()

macro(with_umap target)
    target_include_directories(${target} PUBLIC ${UMAP_INCLUDE_DIR})
    target_link_libraries(${target} ${LIBUMAP})
    target_compile_definitions(${target} PUBLIC METALL_USE_UMAP)
endmacro()

# ---------- MPI ---------- #
find_package(MPI)

# ---------- OpenMP ---------- #
find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else ()
    MESSAGE(WARNING "OpenMP is not found. Use single thread in adjacency_list.")
endif ()


# ---------- Executable ---------- #
include_directories(include)

if (Boost_FOUND)
    if (MPI_CXX_FOUND)
        add_executable(miniVite main.cpp)
        target_link_libraries(miniVite MPI::MPI_CXX)
        target_include_directories(miniVite PUBLIC ${Boost_INCLUDE_DIRS})
        if (USE_METALL)
            with_metall(miniVite)
            if (USE_UMAP)
                with_umap(miniVite)
            endif()
        endif()
    else ()
        message(STATUS "Will skip building the MPI programs")
    endif ()
endif ()