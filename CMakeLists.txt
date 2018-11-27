# Nitin A. Gawande, PNNL
# Nov 27, 2018

PROJECT( dspl ) 
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

cmake_minimum_required( VERSION 3.8 )

# Include cmake modules
list( APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" )

# Project options
option( BUILD_SHARED_LIBS
        "Build shared libraries instead of static libraries."
        OFF )
option( OPTION_SELF_CONTAINED
        "Create a self-contained install with all dependencies."
        OFF )
option( OPTION_BUILD_TESTS
        "Build tests."
        ON )
option( OPTION_BUILD_DOCS
        "Build documentation."
        OFF )
option( OPTION_BUILD_EXAMPLES
        "Build examples."
        OFF )

set( CMAKE_CXX_STANDARD 11 )

#set(CMAKE_C_COMPILER "gcc-8")
#set(CMAKE_CXX_COMPILER "g++-8")
#set(CMAKE_C_COMPILER "/share/apps/gcc/7.1/bin/gcc")
#set(CMAKE_CXX_COMPILER "/share/apps/gcc/7.1/bin/g++")

#set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp -O3" )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11 -fopenmp -O0 -DCHECK_NUM_EDGES -DPRINT_EXTRA_NEDGES" )

find_package(OpenMP)
if (OPENMP_FOUND)
    #set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(MPI REQUIRED)
set(CMAKE_CXX_COMPILER mpicxx)

include_directories(${MPI_INCLUDE_PATH})

add_executable( dspl main.cpp )
#add_executable(hello hello.cpp)
target_link_libraries( dspl ${MPI_LIBRARIES} )

if(MPI_COMPILE_FLAGS)
  set_target_properties( dspl PROPERTIES
    COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties( dspl PROPERTIES
    LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

install( TARGETS
  dspl
  DESTINATION bin
)




