CXX = mpic++
# use -xmic-avx512 instead of -xHost for Intel Xeon Phi platforms
OPTFLAGS = -I. -O3 -fopenmp -DPRINT_DIST_STATS #-DPRINT_EXTRA_NEDGES #-DUSE_MPI_RMA -DUSE_MPI_ACCUMULATE #-DUSE_32_BIT_GRAPH #-DDEBUG_PRINTF #-DUSE_MPI_RMA #-DPRINT_LCG_DOUBLE_LOHI_RANDOM_NUMBERS#-DUSE_MPI_RMA #-DPRINT_LCG_DOUBLE_RANDOM_NUMBERS #-DPRINT_RANDOM_XY_COORD
#-DUSE_MPI_SENDRECV
#-DUSE_MPI_COLLECTIVES
# use export ASAN_OPTIONS=verbosity=1 to check ASAN output
LDFLAGS = -fopenmp
SNTFLAGS = -std=c++11 -openmp -fsanitize=address -O1 -fno-omit-frame-pointer
CXXFLAGS = -std=c++11 -g $(OPTFLAGS)

# metall requires boost libraries and a C++17 compliant compiler
USE_METALL_DSTORE=1
ifeq ($(USE_METALL_DSTORE),1)
    METALL_PATH=./metall/include
    BOOSTINC_PATH=/usr/local/include
    CXXFLAGS += -std=c++17 -lstdc++fs -DUSE_METALL_DSTORE -I$(METALL_PATH) -I$(BOOSTINC_PATH) 
endif

SRC = main.cpp
TARGET = miniVite

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

.PHONY: clean

clean:
	rm -rf *~ *.dSYM *.o $(TARGET)
