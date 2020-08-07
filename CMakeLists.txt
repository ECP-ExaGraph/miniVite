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

# ---------- filesystem ---------- #
include(include_cxx_filesystem_library)
include_cxx_filesystem_library()

# ---------- Boost ---------- #
find_package(Boost 1.64)

# ---------- Metall ---------- #
if (Boost_FOUND)
    if (METALL_ROOT)
        include_directories(${METALL_ROOT}/include)
        add_definitions(-DUSE_METALL_DSTORE)
    endif()
endif()

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

if (Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    include_directories(include)
    if (MPI_CXX_FOUND)
        add_executable(miniVite main.cpp)
        target_link_libraries(miniVite PUBLIC MPI::MPI_CXX)
    else ()
        message(STATUS "Will skip building the MPI programs")
    endif ()
endif ()