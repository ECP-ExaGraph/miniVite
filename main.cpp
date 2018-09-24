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

static std::string inputFileName;
static int me, nprocs;
static int ranksPerNode = 1;
static GraphElem nvRGG = 0;
static bool generateGraph = false;
static int randomEdgePercent = 0;
static bool randomNumberLCG = false;
static bool isUnitEdgeWeight = true;
static double threshold = 1.0E-6;

// parse command line parameters
static void parseCommandLine(const int argc, char * const argv[]);

int main(int argc, char *argv[])
{
  double t0, t1, t2, t3, ti = 0.0;
  
  MPI_Init(&argc, &argv);

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
      //g->print(false);

      if (me == 0) {
          std::cout << "**********************************************************************" << std::endl;
          std::cout << "Generated Random Geometric Graph with d: " << gr.get_d() << std::endl;
          const GraphElem nv = g->get_nv();
          const GraphElem ne = g->get_ne();
          std::cout << "Number of vertices: " << nv << std::endl;
          std::cout << "Number of edges: " << ne << std::endl;
          //std::cout << "Sparsity: "<< (double)((double)nv / (double)(nvRGG*nvRGG))*100.0 <<"%"<< std::endl;
          std::cout << "Average degree: " << (ne / nv) << std::endl;
      }
      
      MPI_Barrier(MPI_COMM_WORLD);
  }
  else { // read input graph
      BinaryEdgeList rm;
      g = rm.read(me, nprocs, ranksPerNode, inputFileName);
      //g->print();
  }

  assert(g != nullptr);

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

  double currMod = -1.0;
  double prevMod = -1.0;
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
  
  if(me == 0) {
      std::cout << "Modularity: " << currMod << ", Iterations: " 
          << iters << ", Time (in s): "<<t0-t1<< std::endl;

      std::cout << "**********************************************************************" << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);

  double tot_time = 0.0;
  MPI_Reduce(&total, &tot_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  
  delete g;
  destroyCommunityMPIType();

  MPI_Finalize();

  return 0;
} // main

void parseCommandLine(const int argc, char * const argv[])
{
  int ret;

  while ((ret = getopt(argc, argv, "f:r:t:n:wlp:")) != -1) {
    switch (ret) {
    case 'f':
      inputFileName.assign(optarg);
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
      randomEdgePercent = atoi(optarg);
      break;
    default:
      assert(0 && "Should not reach here!!");
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
   
  if (me == 0 && !generateGraph && randomEdgePercent) {
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
