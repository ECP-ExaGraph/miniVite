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

#pragma once
#ifndef DSPL_HPP
#define DSPL_HPP

#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <list>
#include <numeric>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <map>

#include <mpi.h>
#include <omp.h>

#include "graph.hpp"
#include "utils.hpp"

struct Comm {
  GraphElem size;
  GraphWeight degree;

  Comm() : size(0), degree(0.0) {};
};

struct CommInfo {
    GraphElem community;
    GraphElem size;
    GraphWeight degree;
};

const int SizeTag           = 1;
const int VertexTag         = 2;
const int CommunityTag      = 3;
const int CommunitySizeTag  = 4;
const int CommunityDataTag  = 5;

static MPI_Datatype commType;

void distSumVertexDegree(const Graph &g, std::vector<GraphWeight> &vDegree, std::vector<Comm> &localCinfo)
{
  const GraphElem nv = g.get_lnv();

#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for default(shared), shared(g, vDegree, localCinfo), schedule(runtime)
#else
#pragma omp parallel for default(shared), shared(g, vDegree, localCinfo), schedule(guided)
#endif
  for (GraphElem i = 0; i < nv; i++) {
    GraphElem e0, e1;
    GraphWeight tw = 0.0;

    g.edge_range(i, e0, e1);

    for (GraphElem k = e0; k < e1; k++) {
      const Edge &edge = g.get_edge(k);
      tw += edge.weight_;
    }

    vDegree[i] = tw;
   
    localCinfo[i].degree = tw;
    localCinfo[i].size = 1L;
  }
} // distSumVertexDegree

GraphWeight distCalcConstantForSecondTerm(const std::vector<GraphWeight> &vDegree, MPI_Comm gcomm)
{
  GraphWeight totalEdgeWeightTwice = 0.0;
  GraphWeight localWeight = 0.0;
  int me = -1;

  const size_t vsz = vDegree.size();

#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for default(shared), shared(vDegree), reduction(+: localWeight) schedule(runtime)
#else
#pragma omp parallel for default(shared), shared(vDegree), reduction(+: localWeight) schedule(static)
#endif  
  for (GraphElem i = 0; i < vsz; i++)
    localWeight += vDegree[i]; // Local reduction

  // Global reduction
  MPI_Allreduce(&localWeight, &totalEdgeWeightTwice, 1, 
          MPI_WEIGHT_TYPE, MPI_SUM, gcomm);

  return (1.0 / static_cast<GraphWeight>(totalEdgeWeightTwice));
} // distCalcConstantForSecondTerm

void distInitComm(std::vector<GraphElem> &pastComm, std::vector<GraphElem> &currComm, const GraphElem base)
{
  const size_t csz = currComm.size();

#ifdef DEBUG_PRINTF  
  assert(csz == pastComm.size());
#endif

#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for default(shared), shared(pastComm, currComm), firstprivate(base), schedule(runtime)
#else
#pragma omp parallel for default(shared), shared(pastComm, currComm), firstprivate(base), schedule(static)
#endif
  for (GraphElem i = 0L; i < csz; i++) {
    pastComm[i] = i + base;
    currComm[i] = i + base;
  }
} // distInitComm

void distInitLouvain(const Graph &dg, std::vector<GraphElem> &pastComm, 
        std::vector<GraphElem> &currComm, std::vector<GraphWeight> &vDegree, 
        std::vector<GraphWeight> &clusterWeight, std::vector<Comm> &localCinfo, 
        std::vector<Comm> &localCupdate, GraphWeight &constantForSecondTerm,
        const int me)
{
  const GraphElem base = dg.get_base(me);
  const GraphElem nv = dg.get_lnv();
  MPI_Comm gcomm = dg.get_comm();

  vDegree.resize(nv);
  pastComm.resize(nv);
  currComm.resize(nv);
  clusterWeight.resize(nv);
  localCinfo.resize(nv);
  localCupdate.resize(nv);
 
  distSumVertexDegree(dg, vDegree, localCinfo);
  constantForSecondTerm = distCalcConstantForSecondTerm(vDegree, gcomm);

  distInitComm(pastComm, currComm, base);
} // distInitLouvain

GraphElem distGetMaxIndex(const std::unordered_map<GraphElem, GraphElem> &clmap, const std::vector<GraphWeight> &counter,
			  const GraphWeight selfLoop, const std::vector<Comm> &localCinfo, 
			  const std::map<GraphElem,Comm> &remoteCinfo, const GraphWeight vDegree, 
                          const GraphElem currSize, const GraphWeight currDegree, const GraphElem currComm,
			  const GraphElem base, const GraphElem bound, const GraphWeight constant)
{
  std::unordered_map<GraphElem, GraphElem>::const_iterator storedAlready;
  GraphElem maxIndex = currComm;
  GraphWeight curGain = 0.0, maxGain = 0.0;
  GraphWeight eix = static_cast<GraphWeight>(counter[0]) - static_cast<GraphWeight>(selfLoop);

  GraphWeight ax = currDegree - vDegree;
  GraphWeight eiy = 0.0, ay = 0.0;

  GraphElem maxSize = currSize; 
  GraphElem size = 0;

  storedAlready = clmap.begin();
#ifdef DEBUG_PRINTF  
  assert(storedAlready != clmap.end());
#endif
  do {
      if (currComm != storedAlready->first) {

          // is_local, direct access local info
          if ((storedAlready->first >= base) && (storedAlready->first < bound)) {
              ay = localCinfo[storedAlready->first-base].degree;
              size = localCinfo[storedAlready->first - base].size;   
          }
          else {
              // is_remote, lookup map
              std::map<GraphElem,Comm>::const_iterator citer = remoteCinfo.find(storedAlready->first);
              ay = citer->second.degree;
              size = citer->second.size; 
          }

          eiy = counter[storedAlready->second];

          curGain = 2.0 * (eiy - eix) - 2.0 * vDegree * (ay - ax) * constant;

          if ((curGain > maxGain) ||
                  ((curGain == maxGain) && (curGain != 0.0) && (storedAlready->first < maxIndex))) {
              maxGain = curGain;
              maxIndex = storedAlready->first;
              maxSize = size;
          }
      }
      storedAlready++;
  } while (storedAlready != clmap.end());

  if ((maxSize == 1) && (currSize == 1) && (maxIndex > currComm))
    maxIndex = currComm;

  return maxIndex;
} // distGetMaxIndex

GraphWeight distBuildLocalMapCounter(const GraphElem e0, const GraphElem e1, std::unordered_map<GraphElem, GraphElem> &clmap, 
				   std::vector<GraphWeight> &counter, const Graph &g, 
                                   const std::vector<GraphElem> &currComm, 
                                   const std::unordered_map<GraphElem, GraphElem> &remoteComm,
	                           const GraphElem vertex, const GraphElem base, const GraphElem bound)
{
  GraphElem numUniqueClusters = 1L;
  GraphWeight selfLoop = 0;
  std::unordered_map<GraphElem, GraphElem>::const_iterator storedAlready;

  for (GraphElem j = e0; j < e1; j++) {
        
    const Edge &edge = g.get_edge(j);
    const GraphElem &tail_ = edge.tail_;
    const GraphWeight &weight = edge.weight_;
    GraphElem tcomm;

    if (tail_ == vertex + base)
      selfLoop += weight;

    // is_local, direct access local std::vector<GraphElem>
    if ((tail_ >= base) && (tail_ < bound))
      tcomm = currComm[tail_ - base];
    else { // is_remote, lookup map
      std::unordered_map<GraphElem, GraphElem>::const_iterator iter = remoteComm.find(tail_);

#ifdef DEBUG_PRINTF  
      assert(iter != remoteComm.end());
#endif
      tcomm = iter->second;
    }

    storedAlready = clmap.find(tcomm);
    
    if (storedAlready != clmap.end())
      counter[storedAlready->second] += weight;
    else {
        clmap.insert(std::unordered_map<GraphElem, GraphElem>::value_type(tcomm, numUniqueClusters));
        counter.push_back(weight);
        numUniqueClusters++;
    }
  }

  return selfLoop;
} // distBuildLocalMapCounter

void distExecuteLouvainIteration(const GraphElem i, const Graph &dg, const std::vector<GraphElem> &currComm,
				 std::vector<GraphElem> &targetComm, const std::vector<GraphWeight> &vDegree,
                                 std::vector<Comm> &localCinfo, std::vector<Comm> &localCupdate,
				 const std::unordered_map<GraphElem, GraphElem> &remoteComm, 
                                 const std::map<GraphElem,Comm> &remoteCinfo, 
                                 std::map<GraphElem,Comm> &remoteCupdate, const GraphWeight constantForSecondTerm,
                                 std::vector<GraphWeight> &clusterWeight, const int me)
{
  GraphElem localTarget = -1;
  GraphElem e0, e1, selfLoop = 0;
  std::unordered_map<GraphElem, GraphElem> clmap;
  std::vector<GraphWeight> counter;

  const GraphElem base = dg.get_base(me), bound = dg.get_bound(me);
  const GraphElem cc = currComm[i];
  GraphWeight ccDegree;
  GraphElem ccSize;  
  bool currCommIsLocal = false; 
  bool targetCommIsLocal = false;

  // Current Community is local
  if (cc >= base && cc < bound) {
	ccDegree=localCinfo[cc-base].degree;
        ccSize=localCinfo[cc-base].size;
        currCommIsLocal=true;
  } else {
  // is remote
        std::map<GraphElem,Comm>::const_iterator citer = remoteCinfo.find(cc);
	ccDegree = citer->second.degree;
 	ccSize = citer->second.size;
	currCommIsLocal=false;
  }

  dg.edge_range(i, e0, e1);

  if (e0 != e1) {
    clmap.insert(std::unordered_map<GraphElem, GraphElem>::value_type(cc, 0));
    counter.push_back(0.0);

    selfLoop =  distBuildLocalMapCounter(e0, e1, clmap, counter, dg, 
                    currComm, remoteComm, i, base, bound);

    clusterWeight[i] += counter[0];

    localTarget = distGetMaxIndex(clmap, counter, selfLoop, localCinfo, remoteCinfo, 
                    vDegree[i], ccSize, ccDegree, cc, base, bound, constantForSecondTerm);
  }
  else
    localTarget = cc;

   // is the Target Local?
   if (localTarget >= base && localTarget < bound)
      targetCommIsLocal = true;
  
  // current and target comm are local - atomic updates to vectors
  if ((localTarget != cc) && (localTarget != -1) && currCommIsLocal && targetCommIsLocal) {
        
#ifdef DEBUG_PRINTF  
        assert( base < localTarget < bound);
        assert( base < cc < bound);
	assert( cc - base < localCupdate.size()); 	
	assert( localTarget - base < localCupdate.size()); 	
#endif
        #pragma omp atomic update
        localCupdate[localTarget-base].degree += vDegree[i];
        #pragma omp atomic update
        localCupdate[localTarget-base].size++;
        #pragma omp atomic update
        localCupdate[cc-base].degree -= vDegree[i];
        #pragma omp atomic update
        localCupdate[cc-base].size--;
     }	

  // current is local, target is not - do atomic on local, accumulate in Maps for remote
  if ((localTarget != cc) && (localTarget != -1) && currCommIsLocal && !targetCommIsLocal) {
        #pragma omp atomic update
        localCupdate[cc-base].degree -= vDegree[i];
        #pragma omp atomic update
        localCupdate[cc-base].size--;
 
        // search target!     
        std::map<GraphElem,Comm>::iterator iter=remoteCupdate.find(localTarget);
 
        #pragma omp atomic update
        iter->second.degree += vDegree[i];
        #pragma omp atomic update
        iter->second.size++;
  }
        
   // current is remote, target is local - accumulate for current, atomic on local
   if ((localTarget != cc) && (localTarget != -1) && !currCommIsLocal && targetCommIsLocal) {
        #pragma omp atomic update
        localCupdate[localTarget-base].degree += vDegree[i];
        #pragma omp atomic update
        localCupdate[localTarget-base].size++;
       
        // search current 
        std::map<GraphElem,Comm>::iterator iter=remoteCupdate.find(cc);
  
        #pragma omp atomic update
        iter->second.degree -= vDegree[i];
        #pragma omp atomic update
        iter->second.size--;
   }
                    
   // current and target are remote - accumulate for both
   if ((localTarget != cc) && (localTarget != -1) && !currCommIsLocal && !targetCommIsLocal) {
       
        // search current 
        std::map<GraphElem,Comm>::iterator iter = remoteCupdate.find(cc);
  
        #pragma omp atomic update
        iter->second.degree -= vDegree[i];
        #pragma omp atomic update
        iter->second.size--;
   
        // search target
        iter=remoteCupdate.find(localTarget);
  
        #pragma omp atomic update
        iter->second.degree += vDegree[i];
        #pragma omp atomic update
        iter->second.size++;
   }

#ifdef DEBUG_PRINTF  
  assert(localTarget != -1);
#endif
  targetComm[i] = localTarget;
} // distExecuteLouvainIteration

GraphWeight distComputeModularity(const Graph &g, std::vector<Comm> &localCinfo,
			     const std::vector<GraphWeight> &clusterWeight,
			     const GraphWeight constantForSecondTerm,
			     const int me)
{
  const GraphElem nv = g.get_lnv();
  MPI_Comm gcomm = g.get_comm();

  GraphWeight le_la_xx[2];
  GraphWeight e_a_xx[2] = {0.0, 0.0};
  GraphWeight le_xx = 0.0, la2_x = 0.0;

#ifdef DEBUG_PRINTF  
  assert((clusterWeight.size() == nv));
#endif

#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for default(shared), shared(clusterWeight, localCinfo), \
  reduction(+: le_xx), reduction(+: la2_x) schedule(runtime)
#else
#pragma omp parallel for default(shared), shared(clusterWeight, localCinfo), \
  reduction(+: le_xx), reduction(+: la2_x) schedule(static)
#endif
  for (GraphElem i = 0L; i < nv; i++) {
    le_xx += clusterWeight[i];
    la2_x += static_cast<GraphWeight>(localCinfo[i].degree) * static_cast<GraphWeight>(localCinfo[i].degree); 
  } 
  le_la_xx[0] = le_xx;
  le_la_xx[1] = la2_x;

#ifdef DEBUG_PRINTF  
  const double t0 = MPI_Wtime();
#endif

  MPI_Allreduce(le_la_xx, e_a_xx, 2, MPI_WEIGHT_TYPE, MPI_SUM, gcomm);

#ifdef DEBUG_PRINTF  
  const double t1 = MPI_Wtime();
#endif

  GraphWeight currMod = std::fabs((e_a_xx[0] * constantForSecondTerm) - 
      (e_a_xx[1] * constantForSecondTerm * constantForSecondTerm));
#ifdef DEBUG_PRINTF  
  std::cout << "[" << me << "]le_xx: " << le_xx << ", la2_x: " << la2_x << std::endl;
  std::cout << "[" << me << "]e_xx: " << e_a_xx[0] << ", a2_x: " << e_a_xx[1] << ", currMod: " << currMod << std::endl;
  std::cout << "[" << me << "]Reduction time: " << (t1 - t0) << std::endl;
#endif

  return currMod;
} // distComputeModularity

void distUpdateLocalCinfo(std::vector<Comm> &localCinfo, const std::vector<Comm> &localCupdate)
{
    size_t csz = localCinfo.size();

#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp for schedule(runtime)
#else
#pragma omp for schedule(static)
#endif
    for (GraphElem i = 0L; i < csz; i++) {
        localCinfo[i].size += localCupdate[i].size;
        localCinfo[i].degree += localCupdate[i].degree;
    }
}

void distCleanCWandCU(const GraphElem nv, std::vector<GraphWeight> &clusterWeight,
        std::vector<Comm> &localCupdate)
{
#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp for schedule(runtime)
#else
#pragma omp for schedule(static)
#endif
    for (GraphElem i = 0L; i < nv; i++) {
        clusterWeight[i] = 0;
        localCupdate[i].degree = 0;
        localCupdate[i].size = 0;
    }
} // distCleanCWandCU

#if defined(USE_MPI_RMA)
void fillRemoteCommunities(const Graph &dg, const int me, const int nprocs,
        const size_t &ssz, const size_t &rsz, const std::vector<GraphElem> &ssizes, 
        const std::vector<GraphElem> &rsizes, const std::vector<GraphElem> &svdata, 
        const std::vector<GraphElem> &rvdata, const std::vector<GraphElem> &currComm, 
        const std::vector<Comm> &localCinfo, std::map<GraphElem,Comm> &remoteCinfo, 
        std::unordered_map<GraphElem, GraphElem> &remoteComm, std::map<GraphElem,Comm> &remoteCupdate, 
        const MPI_Win &commwin, const std::vector<GraphElem> &disp)
#else
void fillRemoteCommunities(const Graph &dg, const int me, const int nprocs,
        const size_t &ssz, const size_t &rsz, const std::vector<GraphElem> &ssizes, 
        const std::vector<GraphElem> &rsizes, const std::vector<GraphElem> &svdata, 
        const std::vector<GraphElem> &rvdata, const std::vector<GraphElem> &currComm, 
        const std::vector<Comm> &localCinfo, std::map<GraphElem,Comm> &remoteCinfo, 
        std::unordered_map<GraphElem, GraphElem> &remoteComm, std::map<GraphElem,Comm> &remoteCupdate)
#endif
{
#if defined(USE_MPI_RMA)
    std::vector<GraphElem> scdata(ssz);
#else
    std::vector<GraphElem> rcdata(rsz), scdata(ssz);
#endif
  GraphElem spos, rpos;
#if defined(REPLACE_STL_UOSET_WITH_VECTOR)
  std::vector< std::vector< GraphElem > > rcinfo(nprocs);
#else
  std::vector<std::unordered_set<GraphElem> > rcinfo(nprocs);
#endif

#if defined(USE_MPI_SENDRECV)
#else
  std::vector<MPI_Request> rreqs(nprocs), sreqs(nprocs);
#endif

#ifdef DEBUG_PRINTF  
  double t0, t1, ta = 0.0;
#endif

#if defined(USE_MPI_RMA) && !defined(USE_MPI_ACCUMULATE)
  int num_comm_procs;
#endif

#if defined(USE_MPI_RMA) && !defined(USE_MPI_ACCUMULATE)
  spos = 0;
  rpos = 0;
  std::vector<int> comm_proc(nprocs);
  std::vector<int> comm_proc_buf_disp(nprocs);
  
  /* Initialize all to -1 (unsure if necessary) */
  for (int i = 0; i < nprocs; i++) {
      comm_proc[i] = -1;
      comm_proc_buf_disp[i] = -1;
  }
  
  num_comm_procs = 0;
  for (int i = 0; i < nprocs; i++) {
      if ((i != me) && (ssizes[i] > 0)) {
          comm_proc[num_comm_procs] = i;
          comm_proc_buf_disp[num_comm_procs] = spos;
          num_comm_procs++;
      }
      spos += ssizes[i];
      rpos += rsizes[i];
  }
#endif

  const GraphElem base = dg.get_base(me), bound = dg.get_bound(me);
  const GraphElem nv = dg.get_lnv();
  MPI_Comm gcomm = dg.get_comm();

  // Collects Communities of local vertices for remote nodes
#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for shared(svdata, scdata, currComm) schedule(runtime)
#else
#pragma omp parallel for shared(svdata, scdata, currComm) schedule(static)
#endif
  for (GraphElem i = 0; i < ssz; i++) {
    const GraphElem vertex = svdata[i];
#ifdef DEBUG_PRINTF  
    assert((vertex >= base) && (vertex < bound));
#endif
    const GraphElem comm = currComm[vertex - base];
    scdata[i] = comm;
  }

  std::vector<GraphElem> rcsizes(nprocs), scsizes(nprocs);
  std::vector<CommInfo> sinfo, rinfo;

#ifdef DEBUG_PRINTF  
  t0 = MPI_Wtime();
#endif
#if !defined(USE_MPI_RMA) || defined(USE_MPI_ACCUMULATE)
  spos = 0;
  rpos = 0;
#endif
#if defined(USE_MPI_COLLECTIVES)
  std::vector<int> scnts(nprocs), rcnts(nprocs), sdispls(nprocs), rdispls(nprocs);
  for (int i = 0; i < nprocs; i++) {
      scnts[i] = ssizes[i];
      rcnts[i] = rsizes[i];
      sdispls[i] = spos;
      rdispls[i] = rpos;
      spos += scnts[i];
      rpos += rcnts[i];
  }
  scnts[me] = 0;
  rcnts[me] = 0;
  MPI_Alltoallv(scdata.data(), scnts.data(), sdispls.data(), 
          MPI_GRAPH_TYPE, rcdata.data(), rcnts.data(), rdispls.data(), 
          MPI_GRAPH_TYPE, gcomm);
#elif defined(USE_MPI_RMA)
#if defined(USE_MPI_ACCUMULATE)
  for (int i = 0; i < nprocs; i++) {
      if ((i != me) && (ssizes[i] > 0)) {
          MPI_Accumulate(scdata.data() + spos, ssizes[i], MPI_GRAPH_TYPE, i, 
                  disp[i], ssizes[i], MPI_GRAPH_TYPE, MPI_REPLACE, commwin);
      }
      spos += ssizes[i];
      rpos += rsizes[i];
  }
#else
  for (int i = 0; i < num_comm_procs; i++) {
      int target_rank = comm_proc[i];
      MPI_Put(scdata.data() + comm_proc_buf_disp[i], ssizes[target_rank], MPI_GRAPH_TYPE,
              target_rank, disp[target_rank], ssizes[target_rank], MPI_GRAPH_TYPE, commwin);
  }
#endif
#elif defined(USE_MPI_SENDRECV)
  for (int i = 0; i < nprocs; i++) {
      if (i != me)
          MPI_Sendrecv(scdata.data() + spos, ssizes[i], MPI_GRAPH_TYPE, i, CommunityTag, 
                  rcdata.data() + rpos, rsizes[i], MPI_GRAPH_TYPE, i, CommunityTag, 
                  gcomm, MPI_STATUSES_IGNORE);

      spos += ssizes[i];
      rpos += rsizes[i];
  }
#else
  for (int i = 0; i < nprocs; i++) {
    if ((i != me) && (rsizes[i] > 0))
      MPI_Irecv(rcdata.data() + rpos, rsizes[i], MPI_GRAPH_TYPE, i, 
              CommunityTag, gcomm, &rreqs[i]);
    else
      rreqs[i] = MPI_REQUEST_NULL;

    rpos += rsizes[i];
  }
  for (int i = 0; i < nprocs; i++) {
    if ((i != me) && (ssizes[i] > 0))
      MPI_Isend(scdata.data() + spos, ssizes[i], MPI_GRAPH_TYPE, i, 
              CommunityTag, gcomm, &sreqs[i]);
    else
      sreqs[i] = MPI_REQUEST_NULL;

    spos += ssizes[i];
  }

  MPI_Waitall(nprocs, sreqs.data(), MPI_STATUSES_IGNORE);
  MPI_Waitall(nprocs, rreqs.data(), MPI_STATUSES_IGNORE);
#endif
#ifdef DEBUG_PRINTF  
  t1 = MPI_Wtime();
  ta += (t1 - t0);
#endif

  // reserve vectors
#if defined(REPLACE_STL_UOSET_WITH_VECTOR)
  for (GraphElem i = 0; i < nprocs; i++) {
      rcinfo[i].reserve(rpos);
  }
#endif

  // fetch baseptr from MPI window
#if defined(USE_MPI_RMA)
  MPI_Win_flush_all(commwin);
  MPI_Barrier(gcomm);

  GraphElem *rcbuf = nullptr;
  int flag = 0;
  MPI_Win_get_attr(commwin, MPI_WIN_BASE, &rcbuf, &flag);
#endif

  remoteComm.clear();
  for (GraphElem i = 0; i < rpos; i++) {

#if defined(USE_MPI_RMA)
    const GraphElem comm = rcbuf[i];
#else
    const GraphElem comm = rcdata[i];
#endif

    remoteComm.insert(std::unordered_map<GraphElem, GraphElem>::value_type(rvdata[i], comm));
    const int tproc = dg.get_owner(comm);

    if (tproc != me)
#if defined(REPLACE_STL_UOSET_WITH_VECTOR)
      rcinfo[tproc].emplace_back(comm);
#else
      rcinfo[tproc].insert(comm);
#endif
  }

  for (GraphElem i = 0; i < nv; i++) {
    const GraphElem comm = currComm[i];
    const int tproc = dg.get_owner(comm);

    if (tproc != me)
#if defined(REPLACE_STL_UOSET_WITH_VECTOR)
      rcinfo[tproc].emplace_back(comm);
#else
      rcinfo[tproc].insert(comm);
#endif
  }

#ifdef DEBUG_PRINTF  
  t0 = MPI_Wtime();
#endif
  GraphElem stcsz = 0, rtcsz = 0;
  
#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for shared(scsizes, rcinfo) \
  reduction(+:stcsz) schedule(runtime)
#else
#pragma omp parallel for shared(scsizes, rcinfo) \
  reduction(+:stcsz) schedule(static)
#endif
  for (int i = 0; i < nprocs; i++) {
    scsizes[i] = rcinfo[i].size();
    stcsz += scsizes[i];
  }

  MPI_Alltoall(scsizes.data(), 1, MPI_GRAPH_TYPE, rcsizes.data(), 
          1, MPI_GRAPH_TYPE, gcomm);

#ifdef DEBUG_PRINTF  
  t1 = MPI_Wtime();
  ta += (t1 - t0);
#endif

#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for shared(rcsizes) \
  reduction(+:rtcsz) schedule(runtime)
#else
#pragma omp parallel for shared(rcsizes) \
  reduction(+:rtcsz) schedule(static)
#endif
  for (int i = 0; i < nprocs; i++) {
    rtcsz += rcsizes[i];
  }

#ifdef DEBUG_PRINTF  
  std::cout << "[" << me << "]Total communities to receive: " << rtcsz << std::endl;
#endif
#if defined(USE_MPI_COLLECTIVES)
  std::vector<GraphElem> rcomms(rtcsz), scomms(stcsz);
#else
#if defined(REPLACE_STL_UOSET_WITH_VECTOR)
  std::vector<GraphElem> rcomms(rtcsz);
#else
  std::vector<GraphElem> rcomms(rtcsz), scomms(stcsz);
#endif
#endif
  sinfo.resize(rtcsz);
  rinfo.resize(stcsz);

#ifdef DEBUG_PRINTF  
  t0 = MPI_Wtime();
#endif
  spos = 0;
  rpos = 0;
#if defined(USE_MPI_COLLECTIVES)
  for (int i = 0; i < nprocs; i++) {
      if (i != me) {
          std::copy(rcinfo[i].begin(), rcinfo[i].end(), scomms.data() + spos);
      }
      scnts[i] = scsizes[i];
      rcnts[i] = rcsizes[i];
      sdispls[i] = spos;
      rdispls[i] = rpos;
      spos += scnts[i];
      rpos += rcnts[i];
  }
  scnts[me] = 0;
  rcnts[me] = 0;
  MPI_Alltoallv(scomms.data(), scnts.data(), sdispls.data(), 
          MPI_GRAPH_TYPE, rcomms.data(), rcnts.data(), rdispls.data(), 
          MPI_GRAPH_TYPE, gcomm);

  for (int i = 0; i < nprocs; i++) {
      if (i != me) {
#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for default(none), shared(rcsizes, rcomms, localCinfo, sinfo, rdispls), \
          firstprivate(i, base), schedule(runtime) , if(rcsizes[i] >= 1000)
#else
#pragma omp parallel for default(none), shared(rcsizes, rcomms, localCinfo, sinfo, rdispls), \
          firstprivate(i, base), schedule(guided) , if(rcsizes[i] >= 1000)
#endif
          for (GraphElem j = 0; j < rcsizes[i]; j++) {
              const GraphElem comm = rcomms[rdispls[i] + j];
              sinfo[rdispls[i] + j] = {comm, localCinfo[comm-base].size, localCinfo[comm-base].degree};
          }
      }
  }
  
  MPI_Alltoallv(sinfo.data(), rcnts.data(), rdispls.data(), 
          commType, rinfo.data(), scnts.data(), sdispls.data(), 
          commType, gcomm);
#else
#if !defined(USE_MPI_SENDRECV)
  std::vector<MPI_Request> rcreqs(nprocs);
#endif
  for (int i = 0; i < nprocs; i++) {
      if (i != me) {
#if defined(USE_MPI_SENDRECV)
#if defined(REPLACE_STL_UOSET_WITH_VECTOR)
          MPI_Sendrecv(rcinfo[i].data(), scsizes[i], MPI_GRAPH_TYPE, i, CommunityTag, 
                  rcomms.data() + rpos, rcsizes[i], MPI_GRAPH_TYPE, i, CommunityTag, 
                  gcomm, MPI_STATUSES_IGNORE);
#else
          std::copy(rcinfo[i].begin(), rcinfo[i].end(), scomms.data() + spos);
          MPI_Sendrecv(scomms.data() + spos, scsizes[i], MPI_GRAPH_TYPE, i, CommunityTag, 
                  rcomms.data() + rpos, rcsizes[i], MPI_GRAPH_TYPE, i, CommunityTag, 
                  gcomm, MPI_STATUSES_IGNORE);
#endif
#else
          if (rcsizes[i] > 0) {
              MPI_Irecv(rcomms.data() + rpos, rcsizes[i], MPI_GRAPH_TYPE, i, 
                      CommunityTag, gcomm, &rreqs[i]);
          }
          else
              rreqs[i] = MPI_REQUEST_NULL;

          if (scsizes[i] > 0) {
#if defined(REPLACE_STL_UOSET_WITH_VECTOR)
              MPI_Isend(rcinfo[i].data(), scsizes[i], MPI_GRAPH_TYPE, i, 
                      CommunityTag, gcomm, &sreqs[i]);
#else
              std::copy(rcinfo[i].begin(), rcinfo[i].end(), scomms.data() + spos);
              MPI_Isend(scomms.data() + spos, scsizes[i], MPI_GRAPH_TYPE, i, 
                      CommunityTag, gcomm, &sreqs[i]);
#endif
          }
          else
              sreqs[i] = MPI_REQUEST_NULL;
#endif
      }
  else {
#if !defined(USE_MPI_SENDRECV)
          rreqs[i] = MPI_REQUEST_NULL;
          sreqs[i] = MPI_REQUEST_NULL;
#endif
      }
      rpos += rcsizes[i];
      spos += scsizes[i];
  }

  spos = 0;
  rpos = 0;
          
  // poke progress on last isend/irecvs
#if !defined(USE_MPI_COLLECTIVES) && !defined(USE_MPI_SENDRECV) && defined(POKE_PROGRESS_FOR_COMMUNITY_SENDRECV_IN_LOOP)
  int tf = 0, id = 0;
  MPI_Testany(nprocs, sreqs.data(), &id, &tf, MPI_STATUS_IGNORE);
#endif

#if !defined(USE_MPI_COLLECTIVES) && !defined(USE_MPI_SENDRECV) && !defined(POKE_PROGRESS_FOR_COMMUNITY_SENDRECV_IN_LOOP)
  MPI_Waitall(nprocs, sreqs.data(), MPI_STATUSES_IGNORE);
  MPI_Waitall(nprocs, rreqs.data(), MPI_STATUSES_IGNORE);
#endif

  for (int i = 0; i < nprocs; i++) {
      if (i != me) {
#if defined(USE_MPI_SENDRECV)
#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for default(none), shared(rcsizes, rcomms, localCinfo, sinfo), \
          firstprivate(i, rpos, base), schedule(runtime) , if(rcsizes[i] >= 1000)

#else
#pragma omp parallel for default(none), shared(rcsizes, rcomms, localCinfo, sinfo), \
          firstprivate(i, rpos, base), schedule(guided) , if(rcsizes[i] >= 1000)
#endif
          for (GraphElem j = 0; j < rcsizes[i]; j++) {
              const GraphElem comm = rcomms[rpos + j];
              sinfo[rpos + j] = {comm, localCinfo[comm-base].size, localCinfo[comm-base].degree};
          }
          
          MPI_Sendrecv(sinfo.data() + rpos, rcsizes[i], commType, i, CommunityDataTag, 
                  rinfo.data() + spos, scsizes[i], commType, i, CommunityDataTag, 
                  gcomm, MPI_STATUSES_IGNORE);
#else
          if (scsizes[i] > 0) {
              MPI_Irecv(rinfo.data() + spos, scsizes[i], commType, i, CommunityDataTag, 
                      gcomm, &rcreqs[i]);
          }
          else
              rcreqs[i] = MPI_REQUEST_NULL;

          // poke progress on last isend/irecvs
#if defined(POKE_PROGRESS_FOR_COMMUNITY_SENDRECV_IN_LOOP)
          int flag = 0, done = 0;
          while (!done) {
              MPI_Test(&sreqs[i], &flag, MPI_STATUS_IGNORE);
              MPI_Test(&rreqs[i], &flag, MPI_STATUS_IGNORE);
              if (flag) 
                  done = 1;
          }
#endif

#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for default(shared), shared(rcsizes, rcomms, localCinfo, sinfo), \
          firstprivate(i, rpos, base), schedule(runtime) , if(rcsizes[i] >= 1000)
#else
#pragma omp parallel for default(shared), shared(rcsizes, rcomms, localCinfo, sinfo), \
          firstprivate(i, rpos, base), schedule(guided) , if(rcsizes[i] >= 1000) 
#endif
          for (GraphElem j = 0; j < rcsizes[i]; j++) {
              const GraphElem comm = rcomms[rpos + j];
              sinfo[rpos + j] = {comm, localCinfo[comm-base].size, localCinfo[comm-base].degree};
          }

          if (rcsizes[i] > 0) {
              MPI_Isend(sinfo.data() + rpos, rcsizes[i], commType, i, 
                      CommunityDataTag, gcomm, &sreqs[i]);
          }
          else
              sreqs[i] = MPI_REQUEST_NULL;
#endif
      }
      else {
#if !defined(USE_MPI_SENDRECV)
          rcreqs[i] = MPI_REQUEST_NULL;
          sreqs[i] = MPI_REQUEST_NULL;
#endif
      }
      rpos += rcsizes[i];
      spos += scsizes[i];
  }

#if !defined(USE_MPI_SENDRECV)
  MPI_Waitall(nprocs, sreqs.data(), MPI_STATUSES_IGNORE);
  MPI_Waitall(nprocs, rcreqs.data(), MPI_STATUSES_IGNORE);
#endif

#endif

#ifdef DEBUG_PRINTF  
  t1 = MPI_Wtime();
  ta += (t1 - t0);
#endif

  remoteCinfo.clear();
  remoteCupdate.clear();

  for (GraphElem i = 0; i < stcsz; i++) {
      const GraphElem ccomm = rinfo[i].community;

      Comm comm;

      comm.size = rinfo[i].size;
      comm.degree = rinfo[i].degree;

      remoteCinfo.insert(std::map<GraphElem,Comm>::value_type(ccomm, comm));
      remoteCupdate.insert(std::map<GraphElem,Comm>::value_type(ccomm, Comm()));
  }
} // end fillRemoteCommunities

void createCommunityMPIType()
{
  CommInfo cinfo;

  MPI_Aint begin, community, size, degree;

  MPI_Get_address(&cinfo, &begin);
  MPI_Get_address(&cinfo.community, &community);
  MPI_Get_address(&cinfo.size, &size);
  MPI_Get_address(&cinfo.degree, &degree);

  int blens[] = { 1, 1, 1 };
  MPI_Aint displ[] = { community - begin, size - begin, degree - begin };
  MPI_Datatype types[] = { MPI_GRAPH_TYPE, MPI_GRAPH_TYPE, MPI_WEIGHT_TYPE };

  MPI_Type_create_struct(3, blens, displ, types, &commType);
  MPI_Type_commit(&commType);
} // createCommunityMPIType

void destroyCommunityMPIType()
{
  MPI_Type_free(&commType);
} // destroyCommunityMPIType

void updateRemoteCommunities(const Graph &dg, std::vector<Comm> &localCinfo,
			     const std::map<GraphElem,Comm> &remoteCupdate,
			     const int me, const int nprocs)
{
  const GraphElem base = dg.get_base(me), bound = dg.get_bound(me);
  std::vector<std::vector<CommInfo>> remoteArray(nprocs);
  MPI_Comm gcomm = dg.get_comm();
  
  // FIXME TODO can we use TBB::concurrent_vector instead,
  // to make this parallel; first we have to get rid of maps
  for (std::map<GraphElem,Comm>::const_iterator iter = remoteCupdate.begin(); iter != remoteCupdate.end(); iter++) {
      const GraphElem i = iter->first;
      const Comm &curr = iter->second;

      const int tproc = dg.get_owner(i);

#ifdef DEBUG_PRINTF  
      assert(tproc != me);
#endif
      CommInfo rcinfo;

      rcinfo.community = i;
      rcinfo.size = curr.size;
      rcinfo.degree = curr.degree;

      remoteArray[tproc].push_back(rcinfo);
  }

  std::vector<GraphElem> send_sz(nprocs), recv_sz(nprocs);

#ifdef DEBUG_PRINTF  
  GraphWeight tc = 0.0;
  const double t0 = MPI_Wtime();
#endif

#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for schedule(runtime)
#else
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < nprocs; i++) {
    send_sz[i] = remoteArray[i].size();
  }

  MPI_Alltoall(send_sz.data(), 1, MPI_GRAPH_TYPE, recv_sz.data(), 
          1, MPI_GRAPH_TYPE, gcomm);

#ifdef DEBUG_PRINTF  
  const double t1 = MPI_Wtime();
  tc += (t1 - t0);
#endif

  GraphElem rcnt = 0, scnt = 0;
#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for shared(recv_sz, send_sz) \
  reduction(+:rcnt, scnt) schedule(runtime)
#else
#pragma omp parallel for shared(recv_sz, send_sz) \
  reduction(+:rcnt, scnt) schedule(static)
#endif
  for (int i = 0; i < nprocs; i++) {
    rcnt += recv_sz[i];
    scnt += send_sz[i];
  }
#ifdef DEBUG_PRINTF  
  std::cout << "[" << me << "]Total number of remote communities to update: " << scnt << std::endl;
#endif

  GraphElem currPos = 0;
  std::vector<CommInfo> rdata(rcnt);

#ifdef DEBUG_PRINTF  
  const double t2 = MPI_Wtime();
#endif
#if defined(USE_MPI_SENDRECV)
  for (int i = 0; i < nprocs; i++) {
      if (i != me)
          MPI_Sendrecv(remoteArray[i].data(), send_sz[i], commType, i, CommunityDataTag, 
                  rdata.data() + currPos, recv_sz[i], commType, i, CommunityDataTag, 
                  gcomm, MPI_STATUSES_IGNORE);

      currPos += recv_sz[i];
  }
#else
  std::vector<MPI_Request> sreqs(nprocs), rreqs(nprocs);
  for (int i = 0; i < nprocs; i++) {
    if ((i != me) && (recv_sz[i] > 0))
      MPI_Irecv(rdata.data() + currPos, recv_sz[i], commType, i, 
              CommunityDataTag, gcomm, &rreqs[i]);
    else
      rreqs[i] = MPI_REQUEST_NULL;

    currPos += recv_sz[i];
  }

  for (int i = 0; i < nprocs; i++) {
    if ((i != me) && (send_sz[i] > 0))
      MPI_Isend(remoteArray[i].data(), send_sz[i], commType, i, 
              CommunityDataTag, gcomm, &sreqs[i]);
    else
      sreqs[i] = MPI_REQUEST_NULL;
  }

  MPI_Waitall(nprocs, sreqs.data(), MPI_STATUSES_IGNORE);
  MPI_Waitall(nprocs, rreqs.data(), MPI_STATUSES_IGNORE);
#endif
#ifdef DEBUG_PRINTF  
  const double t3 = MPI_Wtime();
  std::cout << "[" << me << "]Update remote community MPI time: " << (t3 - t2) << std::endl;
#endif

#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for shared(rdata, localCinfo) schedule(runtime)
#else
#pragma omp parallel for shared(rdata, localCinfo) schedule(dynamic)
#endif
  for (GraphElem i = 0; i < rcnt; i++) {
    const CommInfo &curr = rdata[i];

#ifdef DEBUG_PRINTF  
    assert(dg.get_owner(curr.community) == me);
#endif
    localCinfo[curr.community-base].size += curr.size;
    localCinfo[curr.community-base].degree += curr.degree;
  }
} // updateRemoteCommunities

// initial setup before Louvain iteration begins
#if defined(USE_MPI_RMA)
void exchangeVertexReqs(const Graph &dg, size_t &ssz, size_t &rsz,
        std::vector<GraphElem> &ssizes, std::vector<GraphElem> &rsizes, 
        std::vector<GraphElem> &svdata, std::vector<GraphElem> &rvdata,
        const int me, const int nprocs, MPI_Win &commwin)
#else
void exchangeVertexReqs(const Graph &dg, size_t &ssz, size_t &rsz,
        std::vector<GraphElem> &ssizes, std::vector<GraphElem> &rsizes, 
        std::vector<GraphElem> &svdata, std::vector<GraphElem> &rvdata,
        const int me, const int nprocs)
#endif
{
  const GraphElem base = dg.get_base(me), bound = dg.get_bound(me);
  const GraphElem nv = dg.get_lnv();
  MPI_Comm gcomm = dg.get_comm();

#ifdef USE_OPENMP_LOCK
  std::vector<omp_lock_t> locks(nprocs);
  for (int i = 0; i < nprocs; i++)
    omp_init_lock(&locks[i]);
#endif
  std::vector<std::unordered_set<GraphElem>> parray(nprocs);

#ifdef USE_OPENMP_LOCK
#pragma omp parallel default(shared), shared(dg, locks, parray), firstprivate(me)
#else
#pragma omp parallel default(shared), shared(dg, parray), firstprivate(me)
#endif
  {
#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp for schedule(runtime)
#else
#pragma omp for schedule(guided)
#endif
    for (GraphElem i = 0; i < nv; i++) {
      GraphElem e0, e1;

      dg.edge_range(i, e0, e1);

      for (GraphElem j = e0; j < e1; j++) {
	const Edge &edge = dg.get_edge(j);
	const int tproc = dg.get_owner(edge.tail_);

	if (tproc != me) {
#ifdef USE_OPENMP_LOCK
	  omp_set_lock(&locks[tproc]);
#else
          lock();
#endif
	  parray[tproc].insert(edge.tail_);
#ifdef USE_OPENMP_LOCK
	  omp_unset_lock(&locks[tproc]);
#else
          unlock();
#endif
	}
      }
    }
  }

#ifdef USE_OPENMP_LOCK
  for (int i = 0; i < nprocs; i++) {
    omp_destroy_lock(&locks[i]);
  }
#endif
  
  rsizes.resize(nprocs);
  ssizes.resize(nprocs);
  ssz = 0, rsz = 0;

  int pproc = 0;
  // TODO FIXME parallelize this loop
  for (std::vector<std::unordered_set<GraphElem>>::const_iterator iter = parray.begin(); iter != parray.end(); iter++) {
    ssz += iter->size();
    ssizes[pproc] = iter->size();
    pproc++;
  }

  MPI_Alltoall(ssizes.data(), 1, MPI_GRAPH_TYPE, rsizes.data(), 
          1, MPI_GRAPH_TYPE, gcomm);

  GraphElem rsz_r = 0;
#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for shared(rsizes) \
  reduction(+:rsz_r) schedule(runtime)
#else
#pragma omp parallel for shared(rsizes) \
  reduction(+:rsz_r) schedule(static)
#endif
  for (int i = 0; i < nprocs; i++)
    rsz_r += rsizes[i];
  rsz = rsz_r;
  
  svdata.resize(ssz);
  rvdata.resize(rsz);

  GraphElem cpos = 0, rpos = 0;
  pproc = 0;

#if defined(USE_MPI_COLLECTIVES)
  std::vector<int> scnts(nprocs), rcnts(nprocs), sdispls(nprocs), rdispls(nprocs);
  
  for (std::vector<std::unordered_set<GraphElem>>::const_iterator iter = parray.begin(); iter != parray.end(); iter++) {
      std::copy(iter->begin(), iter->end(), svdata.begin() + cpos);
      
      scnts[pproc] = iter->size();
      rcnts[pproc] = rsizes[pproc];
      sdispls[pproc] = cpos;
      rdispls[pproc] = rpos;
      cpos += iter->size();
      rpos += rcnts[pproc];

      pproc++;
  }

  scnts[me] = 0;
  rcnts[me] = 0;
  MPI_Alltoallv(svdata.data(), scnts.data(), sdispls.data(), 
          MPI_GRAPH_TYPE, rvdata.data(), rcnts.data(), rdispls.data(), 
          MPI_GRAPH_TYPE, gcomm);
#else
  std::vector<MPI_Request> rreqs(nprocs), sreqs(nprocs);
  for (int i = 0; i < nprocs; i++) {
      if ((i != me) && (rsizes[i] > 0))
          MPI_Irecv(rvdata.data() + rpos, rsizes[i], MPI_GRAPH_TYPE, i, 
                  VertexTag, gcomm, &rreqs[i]);
      else
          rreqs[i] = MPI_REQUEST_NULL;

      rpos += rsizes[i];
  }

  for (std::vector<std::unordered_set<GraphElem>>::const_iterator iter = parray.begin(); iter != parray.end(); iter++) {
      std::copy(iter->begin(), iter->end(), svdata.begin() + cpos);

      if ((me != pproc) && (iter->size() > 0))
          MPI_Isend(svdata.data() + cpos, iter->size(), MPI_GRAPH_TYPE, pproc, 
                  VertexTag, gcomm, &sreqs[pproc]);
      else
          sreqs[pproc] = MPI_REQUEST_NULL;

      cpos += iter->size();
      pproc++;
  }

  MPI_Waitall(nprocs, sreqs.data(), MPI_STATUSES_IGNORE);
  MPI_Waitall(nprocs, rreqs.data(), MPI_STATUSES_IGNORE);
#endif

  std::swap(svdata, rvdata);
  std::swap(ssizes, rsizes);
  std::swap(ssz, rsz);

  // create MPI window for communities
#if defined(USE_MPI_RMA)  
  GraphElem *ptr = nullptr;
  MPI_Info info = MPI_INFO_NULL;
#if defined(USE_MPI_ACCUMULATE)
  MPI_Info_create(&info);
  MPI_Info_set(info, "accumulate_ordering", "none");
  MPI_Info_set(info, "accumulate_ops", "same_op");
#endif
  MPI_Win_allocate(rsz*sizeof(GraphElem), sizeof(GraphElem), 
          info, gcomm, &ptr, &commwin);
  MPI_Win_lock_all(MPI_MODE_NOCHECK, commwin);
#endif
} // exchangeVertexReqs

#if defined(USE_MPI_RMA)
GraphWeight distLouvainMethod(const int me, const int nprocs, const Graph &dg,
        size_t &ssz, size_t &rsz, std::vector<GraphElem> &ssizes, std::vector<GraphElem> &rsizes, 
        std::vector<GraphElem> &svdata, std::vector<GraphElem> &rvdata, const GraphWeight lower, 
        const GraphWeight thresh, int &iters, MPI_Win &commwin)
#else
GraphWeight distLouvainMethod(const int me, const int nprocs, const Graph &dg,
        size_t &ssz, size_t &rsz, std::vector<GraphElem> &ssizes, std::vector<GraphElem> &rsizes, 
        std::vector<GraphElem> &svdata, std::vector<GraphElem> &rvdata, const GraphWeight lower, 
        const GraphWeight thresh, int &iters)
#endif
{
  std::vector<GraphElem> pastComm, currComm, targetComm;
  std::vector<GraphWeight> vDegree;
  std::vector<GraphWeight> clusterWeight;
  std::vector<Comm> localCinfo, localCupdate;
 
  std::unordered_map<GraphElem, GraphElem> remoteComm;
  std::map<GraphElem,Comm> remoteCinfo, remoteCupdate;
  
  const GraphElem nv = dg.get_lnv();
  MPI_Comm gcomm = dg.get_comm();

  GraphWeight constantForSecondTerm;
  GraphWeight prevMod = lower;
  GraphWeight currMod = -1.0;
  int numIters = 0;
  
  distInitLouvain(dg, pastComm, currComm, vDegree, clusterWeight, localCinfo, 
          localCupdate, constantForSecondTerm, me);
  targetComm.resize(nv);

#ifdef DEBUG_PRINTF  
  std::cout << "[" << me << "]constantForSecondTerm: " << constantForSecondTerm << std::endl;
  if (me == 0)
      std::cout << "Threshold: " << thresh << std::endl;
#endif
  const GraphElem base = dg.get_base(me), bound = dg.get_bound(me);

#ifdef DEBUG_PRINTF  
  double t0, t1;
  t0 = MPI_Wtime();
#endif

  // setup vertices and communities
#if defined(USE_MPI_RMA)
  exchangeVertexReqs(dg, ssz, rsz, ssizes, rsizes, 
          svdata, rvdata, me, nprocs, commwin);
  
  // store the remote displacements 
  std::vector<GraphElem> disp(nprocs);
  MPI_Exscan(ssizes.data(), (GraphElem*)disp.data(), nprocs, MPI_GRAPH_TYPE, 
          MPI_SUM, gcomm);
#else
  exchangeVertexReqs(dg, ssz, rsz, ssizes, rsizes, 
          svdata, rvdata, me, nprocs);
#endif

#ifdef DEBUG_PRINTF  
  t1 = MPI_Wtime();
  std::cout << "[" << me << "]Initial communication setup time before Louvain iteration (in s): " << (t1 - t0) << std::endl;
#endif
 
  // start Louvain iteration
  while(true) {
#ifdef DEBUG_PRINTF  
    const double t2 = MPI_Wtime();
    if (me == 0)
        std::cout << "Starting Louvain iteration: " << numIters << std::endl;
#endif
    numIters++;

#ifdef DEBUG_PRINTF  
    t0 = MPI_Wtime();
#endif

#if defined(USE_MPI_RMA)
    fillRemoteCommunities(dg, me, nprocs, ssz, rsz, ssizes, 
            rsizes, svdata, rvdata, currComm, localCinfo, 
            remoteCinfo, remoteComm, remoteCupdate, 
            commwin, disp);
#else
    fillRemoteCommunities(dg, me, nprocs, ssz, rsz, ssizes, 
            rsizes, svdata, rvdata, currComm, localCinfo, 
            remoteCinfo, remoteComm, remoteCupdate);
#endif

#ifdef DEBUG_PRINTF  
    t1 = MPI_Wtime();
    std::cout << "[" << me << "]Remote community map size: " << remoteComm.size() << std::endl;
    std::cout << "[" << me << "]Iteration communication time: " << (t1 - t0) << std::endl;
#endif

#ifdef DEBUG_PRINTF  
    t0 = MPI_Wtime();
#endif

#pragma omp parallel default(shared), shared(clusterWeight, localCupdate, currComm, targetComm, \
        vDegree, localCinfo, remoteCinfo, remoteComm, pastComm, dg, remoteCupdate), \
        firstprivate(constantForSecondTerm, me)
    {
        distCleanCWandCU(nv, clusterWeight, localCupdate);

#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp for schedule(runtime)
#else
#pragma omp for schedule(guided) 
#endif
        for (GraphElem i = 0; i < nv; i++) {
            distExecuteLouvainIteration(i, dg, currComm, targetComm, vDegree, localCinfo, 
                    localCupdate, remoteComm, remoteCinfo, remoteCupdate,
                    constantForSecondTerm, clusterWeight, me);
        }
    }

#pragma omp parallel default(none), shared(localCinfo, localCupdate)
    {
        distUpdateLocalCinfo(localCinfo, localCupdate);
    }

    // communicate remote communities
    updateRemoteCommunities(dg, localCinfo, remoteCupdate, me, nprocs);

    // compute modularity
    currMod = distComputeModularity(dg, localCinfo, clusterWeight, constantForSecondTerm, me);

    // exit criteria
    if (currMod - prevMod < thresh)
        break;

    prevMod = currMod;
    if (prevMod < lower)
        prevMod = lower;

#ifdef OMP_SCHEDULE_RUNTIME
#pragma omp parallel for default(shared) \
    shared(pastComm, currComm, targetComm) \
    schedule(runtime)
#else
#pragma omp parallel for default(shared) \
    shared(pastComm, currComm, targetComm) \
    schedule(static)
#endif
    for (GraphElem i = 0; i < nv; i++) {
        GraphElem tmp = pastComm[i];
        pastComm[i] = currComm[i];
        currComm[i] = targetComm[i];
        targetComm[i] = tmp;
    }
  } // end of Louvain iteration

#if defined(USE_MPI_RMA)
  MPI_Win_unlock_all(commwin);
  MPI_Win_free(&commwin);
#endif  

  iters = numIters;

  vDegree.clear();
  pastComm.clear();
  currComm.clear();
  targetComm.clear();
  clusterWeight.clear();
  localCinfo.clear();
  localCupdate.clear();
  
  return prevMod;
} // distLouvainMethod plain

#endif // __DSPL
