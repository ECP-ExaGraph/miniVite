// ***********************************************************************
//
//                              miniVite
//
// ***********************************************************************
//
//       Copyright (2018) Battelle Memorial Institute
//                      All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// ************************************************************************ 


#include <sys/resource.h>
#include <sys/time.h>
#include <unistd.h>

#include <cassert>
#include <cstdlib>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <omp.h>
#include <mpi.h>

#include "dspl.hpp"

// TODO FIXME add options for desired MPI thread-level

static std::string inputFileName;
static int me, nprocs;
static int ranksPerNode = 1;
static GraphElem nvRGG = 0;
static bool generateGraph = false;
static bool readBalanced = false;
static bool showGraph = false;
static GraphWeight randomEdgePercent = 0.0;
static bool randomNumberLCG = false;
static bool isUnitEdgeWeight = true;
static GraphWeight threshold = 1.0E-6;

// parse command line parameters
static void parseCommandLine(const int argc, char * const argv[]);

int main(int argc, char *argv[])
{
  double t0, t1, t2, t3, ti = 0.0;
#ifdef DISABLE_THREAD_MULTIPLE_CHECK
  MPI_Init(&argc, &argv);
#else  
  int max_threads;

  max_threads = omp_get_max_threads();

  if (max_threads > 1) {
      int provided;
      MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
      if (provided < MPI_THREAD_MULTIPLE) {
          std::cerr << "MPI library does not support MPI_THREAD_MULTIPLE." << std::endl;
          MPI_Abort(MPI_COMM_WORLD, -99);
      }
  } else {
      MPI_Init(&argc, &argv);
  }
#endif

  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &me);

  parseCommandLine(argc, argv);

  createCommunityMPIType();
  double td0, td1, td, tdt;

  MPI_Barrier(MPI_COMM_WORLD);
  td0 = MPI_Wtime();

  Graph* g = nullptr;

  // generate graph only supports RGG as of now
  if (generateGraph) { 
      GenerateRGG gr(nvRGG);
      g = gr.generate(randomNumberLCG, isUnitEdgeWeight, randomEdgePercent);
  }
  else { // read input graph
      BinaryEdgeList rm;
      if (readBalanced == true)
          g = rm.read_balanced(me, nprocs, ranksPerNode, inputFileName);
      else
          g = rm.read(me, nprocs, ranksPerNode, inputFileName);
  }

  assert(g != nullptr);
  if (showGraph)
      g->print();

#ifdef PRINT_DIST_STATS 
  g->print_dist_stats();
#endif

  MPI_Barrier(MPI_COMM_WORLD);
#ifdef DEBUG_PRINTF  
  assert(g);
#endif  
  td1 = MPI_Wtime();
  td = td1 - td0;

  MPI_Reduce(&td, &tdt, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
 
  if (me == 0)  {
      if (!generateGraph)
          std::cout << "Time to read input file and create distributed graph (in s): " 
              << (tdt/nprocs) << std::endl;
      else
          std::cout << "Time to generate distributed graph of " 
              << nvRGG << " vertices (in s): " << (tdt/nprocs) << std::endl;
  }

  GraphWeight currMod = -1.0;
  GraphWeight prevMod = -1.0;
  double total = 0.0;

  std::vector<GraphElem> ssizes, rsizes, svdata, rvdata;
#if defined(USE_MPI_RMA)
  MPI_Win commwin;
#endif
  size_t ssz = 0, rsz = 0;
  int iters = 0;
    
  MPI_Barrier(MPI_COMM_WORLD);

  t1 = MPI_Wtime();

#if defined(USE_MPI_RMA)
  currMod = distLouvainMethod(me, nprocs, *g, ssz, rsz, ssizes, rsizes, 
                svdata, rvdata, currMod, threshold, iters, commwin);
#else
  currMod = distLouvainMethod(me, nprocs, *g, ssz, rsz, ssizes, rsizes, 
                svdata, rvdata, currMod, threshold, iters);
#endif
  MPI_Barrier(MPI_COMM_WORLD);
  t0 = MPI_Wtime();
  total = t0 - t1;

  double tot_time = 0.0;
  MPI_Reduce(&total, &tot_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (me == 0) {
      double avgt = (tot_time / nprocs);
      if (!generateGraph) {
        std::cout << "-------------------------------------------------------" << std::endl;
        std::cout << "File: " << inputFileName << std::endl;
        std::cout << "-------------------------------------------------------" << std::endl;
      }
      std::cout << "-------------------------------------------------------" << std::endl;
#ifdef USE_32_BIT_GRAPH
      std::cout << "32-bit datatype" << std::endl;
#else
      std::cout << "64-bit datatype" << std::endl;
#endif
      std::cout << "-------------------------------------------------------" << std::endl;
      std::cout << "Average total time (in s), #Processes: " << avgt << ", " << nprocs << std::endl;
      std::cout << "Modularity, #Iterations: " << currMod << ", " << iters << std::endl;
      std::cout << "MODS (final modularity * average time): " << (currMod * avgt) << std::endl;
      std::cout << "-------------------------------------------------------" << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  delete g;
  destroyCommunityMPIType();

  MPI_Finalize();

  return 0;
} // main

void parseCommandLine(const int argc, char * const argv[])
{
  int ret;

  while ((ret = getopt(argc, argv, "f:br:t:n:wlp:s")) != -1) {
    switch (ret) {
    case 'f':
      inputFileName.assign(optarg);
      break;
    case 'b':
      readBalanced = true;
      break;
    case 'r':
      ranksPerNode = atoi(optarg);
      break;
    case 't':
      threshold = atof(optarg);
      break;
    case 'n':
      nvRGG = atol(optarg);
      if (nvRGG > 0)
          generateGraph = true; 
      break;
    case 'w':
      isUnitEdgeWeight = false;
      break;
    case 'l':
      randomNumberLCG = true;
      break;
    case 'p':
      randomEdgePercent = atof(optarg);
      break;
    case 's':
      showGraph = true;
      break;
    default:
      assert(0 && "Option not recognized!!!");
      break;
    }
  }

  if (me == 0 && (argc == 1)) {
      std::cerr << "Must specify some options." << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -99);
  }
  
  if (me == 0 && !generateGraph && inputFileName.empty()) {
      std::cerr << "Must specify a binary file name with -f or provide parameters for generating a graph." << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -99);
  }
   
  if (me == 0 && !generateGraph && randomNumberLCG) {
      std::cerr << "Must specify -g for graph generation using LCG." << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -99);
  } 
   
  if (me == 0 && !generateGraph && (randomEdgePercent > 0.0)) {
      std::cerr << "Must specify -g for graph generation first to add random edges to it." << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -99);
  } 
  
  if (me == 0 && !generateGraph && !isUnitEdgeWeight) {
      std::cerr << "Must specify -g for graph generation first before setting edge weights." << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -99);
  }
  
  if (me == 0 && generateGraph && ((randomEdgePercent < 0) || (randomEdgePercent >= 100))) {
      std::cerr << "Invalid random edge percentage for generated graph!" << std::endl;
      MPI_Abort(MPI_COMM_WORLD, -99);
  }
} // parseCommandLine
