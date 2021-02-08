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
#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <climits>
#include <array>
#include <unordered_map>

#include <mpi.h>

#include "utils.hpp"

unsigned seed;

struct Edge
{
    GraphElem tail_;
    GraphWeight weight_;
    
    Edge(): tail_(-1), weight_(0.0) {}
};

struct EdgeTuple
{
    GraphElem ij_[2];
    GraphWeight w_;

    EdgeTuple(GraphElem i, GraphElem j, GraphWeight w): 
        ij_{i, j}, w_(w)
    {}
    EdgeTuple(GraphElem i, GraphElem j): 
        ij_{i, j}, w_(1.0) 
    {}
    EdgeTuple(): 
        ij_{-1, -1}, w_(0.0)
    {}
};

// per process graph instance
class Graph
{
    public:
        Graph(): 
            lnv_(-1), lne_(-1), nv_(-1), 
            ne_(-1), comm_(MPI_COMM_WORLD) 
        {
            MPI_Comm_size(comm_, &size_);
            MPI_Comm_rank(comm_, &rank_);
        }
        
        Graph(GraphElem lnv, GraphElem lne, 
                GraphElem nv, GraphElem ne, 
                MPI_Comm comm=MPI_COMM_WORLD): 
            lnv_(lnv), lne_(lne), 
            nv_(nv), ne_(ne), 
            comm_(comm) 
        {
            MPI_Comm_size(comm_, &size_);
            MPI_Comm_rank(comm_, &rank_);

            edge_indices_.resize(lnv_+1, 0);
            edge_list_.resize(lne_); // this is usually populated later

            parts_.resize(size_+1);
            parts_[0] = 0;

            for (GraphElem i = 1; i < size_+1; i++)
                parts_[i] = ((nv_ * i) / size_);  
        }

        ~Graph() 
        {
            edge_list_.clear();
            edge_indices_.clear();
            parts_.clear();
        }
         
        // update vertex partition information
        void repart(std::vector<GraphElem> const& parts)
        { memcpy(parts_.data(), parts.data(), sizeof(GraphElem)*(size_+1)); }

        // TODO FIXME put asserts like the following
        // everywhere function member of Graph class
        void set_edge_index(GraphElem const vertex, GraphElem const e0)
        {
#if defined(DEBUG_BUILD)
            assert((vertex >= 0) && (vertex <= lnv_));
            assert((e0 >= 0) && (e0 <= lne_));
            edge_indices_.at(vertex) = e0;
#else
            edge_indices_[vertex] = e0;
#endif
        } 
        
        void edge_range(GraphElem const vertex, GraphElem& e0, 
                GraphElem& e1) const
        {
            e0 = edge_indices_[vertex];
            e1 = edge_indices_[vertex+1];
        } 

        // collective
        void set_nedges(GraphElem lne) 
        { 
            lne_ = lne; 
            edge_list_.resize(lne_);

            // compute total number of edges
            ne_ = 0;
            MPI_Allreduce(&lne_, &ne_, 1, MPI_GRAPH_TYPE, MPI_SUM, comm_);
        }

        GraphElem get_base(const int rank) const
        { return parts_[rank]; }

        GraphElem get_bound(const int rank) const
        { return parts_[rank+1]; }
        
        GraphElem get_range(const int rank) const
        { return (parts_[rank+1] - parts_[rank] + 1); }

        int get_owner(const GraphElem vertex) const
        {
            const std::vector<GraphElem>::const_iterator iter = 
                std::upper_bound(parts_.begin(), parts_.end(), vertex);

            return (iter - parts_.begin() - 1);
        }

        GraphElem get_lnv() const { return lnv_; }
        GraphElem get_lne() const { return lne_; }
        GraphElem get_nv() const { return nv_; }
        GraphElem get_ne() const { return ne_; }
        MPI_Comm get_comm() const { return comm_; }
       
        // return edge and active info
        // ----------------------------
       
        Edge const& get_edge(GraphElem const index) const
        { return edge_list_[index]; }
         
        Edge& set_edge(GraphElem const index)
        { return edge_list_[index]; }       
        
        // local <--> global index translation
        // -----------------------------------
        GraphElem local_to_global(GraphElem idx)
        { return (idx + get_base(rank_)); }

        GraphElem global_to_local(GraphElem idx)
        { return (idx - get_base(rank_)); }
       
        // w.r.t passed rank
        GraphElem local_to_global(GraphElem idx, int rank)
        { return (idx + get_base(rank)); }

        GraphElem global_to_local(GraphElem idx, int rank)
        { return (idx - get_base(rank)); }
 
        // print edge list (with weights)
        void print(bool print_weight = true) const
        {
            if (lne_ < MAX_PRINT_NEDGE)
            {
                for (int p = 0; p < size_; p++)
                {
                    MPI_Barrier(comm_);
                    if (p == rank_)
                    {
                        std::cout << "###############" << std::endl;
                        std::cout << "Process #" << p << ": " << std::endl;
                        std::cout << "###############" << std::endl;
                        GraphElem base = get_base(p);
                        for (GraphElem i = 0; i < lnv_; i++)
                        {
                            GraphElem e0, e1;
                            edge_range(i, e0, e1);
                            if (print_weight) { // print weights (default)
                                for (GraphElem e = e0; e < e1; e++)
                                {
                                    Edge const& edge = get_edge(e);
                                    std::cout << i+base << " " << edge.tail_ << " " << edge.weight_ << std::endl;
                                }
                            }
                            else { // don't print weights
                                for (GraphElem e = e0; e < e1; e++)
                                {
                                    Edge const& edge = get_edge(e);
                                    std::cout << i+base << " " << edge.tail_ << std::endl;
                                }
                            }
                        }
                        MPI_Barrier(comm_);
                    }
                }
            }
            else
            {
                if (rank_ == 0)
                    std::cout << "Graph size per process is {" << lnv_ << ", " << lne_ << 
                        "}, which will overwhelm STDOUT." << std::endl;
            }
        }
                
        // print statistics about edge distribution
        void print_dist_stats()
        {
            long sumdeg = 0, maxdeg = 0;
            long lne = (long) lne_;

            MPI_Reduce(&lne, &sumdeg, 1, MPI_LONG, MPI_SUM, 0, comm_);
            MPI_Reduce(&lne, &maxdeg, 1, MPI_LONG, MPI_MAX, 0, comm_);

            long my_sq = lne*lne;
            long sum_sq = 0;
            MPI_Reduce(&my_sq, &sum_sq, 1, MPI_LONG, MPI_SUM, 0, comm_);

            double average  = (double) sumdeg / size_;
            double avg_sq   = (double) sum_sq / size_;
            double var      = avg_sq - (average*average);
            double stddev   = sqrt(var);

            MPI_Barrier(comm_);

            if (rank_ == 0)
            {
                std::cout << std::endl;
                std::cout << "-------------------------------------------------------" << std::endl;
                std::cout << "Graph edge distribution characteristics" << std::endl;
                std::cout << "-------------------------------------------------------" << std::endl;
                std::cout << "Number of vertices: " << nv_ << std::endl;
                std::cout << "Number of edges: " << ne_ << std::endl;
                std::cout << "Maximum number of edges: " << maxdeg << std::endl;
                std::cout << "Average number of edges: " << average << std::endl;
                std::cout << "Expected value of X^2: " << avg_sq << std::endl;
                std::cout << "Variance: " << var << std::endl;
                std::cout << "Standard deviation: " << stddev << std::endl;
                std::cout << "-------------------------------------------------------" << std::endl;

            }
        }
        
        // public variables
        std::vector<GraphElem> edge_indices_;
        std::vector<Edge> edge_list_;
    private:
        GraphElem lnv_, lne_, nv_, ne_;
        std::vector<GraphElem> parts_;       
        MPI_Comm comm_; 
        int rank_, size_;
};

// read in binary edge list files
// using MPI I/O
class BinaryEdgeList
{
    public:
        BinaryEdgeList() : 
            M_(-1), N_(-1), 
            M_local_(-1), N_local_(-1), 
            comm_(MPI_COMM_WORLD) 
        {}
        BinaryEdgeList(MPI_Comm comm) : 
            M_(-1), N_(-1), 
            M_local_(-1), N_local_(-1), 
            comm_(comm) 
        {}
        
        // read a file and return a graph
        Graph* read(int me, int nprocs, int ranks_per_node, std::string file)
        {
            int file_open_error;
            MPI_File fh;
            MPI_Status status;

            // specify the number of aggregates
            MPI_Info info;
            MPI_Info_create(&info);
            int naggr = (ranks_per_node > 1) ? (nprocs/ranks_per_node) : ranks_per_node;
            if (naggr >= nprocs)
                naggr = 1;
            std::stringstream tmp_str;
            tmp_str << naggr;
            std::string str = tmp_str.str();
            MPI_Info_set(info, "cb_nodes", str.c_str());

            file_open_error = MPI_File_open(comm_, file.c_str(), MPI_MODE_RDONLY, info, &fh); 
            MPI_Info_free(&info);

            if (file_open_error != MPI_SUCCESS) 
            {
                std::cout << " Error opening file! " << std::endl;
                MPI_Abort(comm_, -99);
            }

            // read the dimensions 
            MPI_File_read_all(fh, &M_, sizeof(GraphElem), MPI_BYTE, &status);
            MPI_File_read_all(fh, &N_, sizeof(GraphElem), MPI_BYTE, &status);
            M_local_ = ((M_*(me + 1)) / nprocs) - ((M_*me) / nprocs); 

            // create local graph
            Graph *g = new Graph(M_local_, 0, M_, N_);

            // Let N = array length and P = number of processors.
            // From j = 0 to P-1,
            // Starting point of array on processor j = floor(N * j / P)
            // Length of array on processor j = floor(N * (j + 1) / P) - floor(N * j / P)

            uint64_t tot_bytes=(M_local_+1)*sizeof(GraphElem);
            MPI_Offset offset = 2*sizeof(GraphElem) + ((M_*me) / nprocs)*sizeof(GraphElem);

            // read in INT_MAX increments if total byte size is > INT_MAX
            
            if (tot_bytes < INT_MAX)
                MPI_File_read_at(fh, offset, &g->edge_indices_[0], tot_bytes, MPI_BYTE, &status);
            else 
            {
                int chunk_bytes=INT_MAX;
                uint8_t *curr_pointer = (uint8_t*) &g->edge_indices_[0];
                uint64_t transf_bytes = 0;

                while (transf_bytes < tot_bytes)
                {
                    MPI_File_read_at(fh, offset, curr_pointer, chunk_bytes, MPI_BYTE, &status);
                    transf_bytes += chunk_bytes;
                    offset += chunk_bytes;
                    curr_pointer += chunk_bytes;

                    if ((tot_bytes - transf_bytes) < INT_MAX)
                        chunk_bytes = tot_bytes - transf_bytes;
                } 
            }    

            N_local_ = g->edge_indices_[M_local_] - g->edge_indices_[0];
            g->set_nedges(N_local_);

            tot_bytes = N_local_*(sizeof(Edge));
            offset = 2*sizeof(GraphElem) + (M_+1)*sizeof(GraphElem) + g->edge_indices_[0]*(sizeof(Edge));

            if (tot_bytes < INT_MAX)
                MPI_File_read_at(fh, offset, &g->edge_list_[0], tot_bytes, MPI_BYTE, &status);
            else 
            {
                int chunk_bytes=INT_MAX;
                uint8_t *curr_pointer = (uint8_t*)&g->edge_list_[0];
                uint64_t transf_bytes = 0;

                while (transf_bytes < tot_bytes)
                {
                    MPI_File_read_at(fh, offset, curr_pointer, chunk_bytes, MPI_BYTE, &status);
                    transf_bytes += chunk_bytes;
                    offset += chunk_bytes;
                    curr_pointer += chunk_bytes;

                    if ((tot_bytes - transf_bytes) < INT_MAX)
                        chunk_bytes = (tot_bytes - transf_bytes);
                } 
            }    

            MPI_File_close(&fh);

            for(GraphElem i=1;  i < M_local_+1; i++)
                g->edge_indices_[i] -= g->edge_indices_[0];   
            g->edge_indices_[0] = 0;

            return g;
        }

        // find a distribution such that every 
        // process own equal number of edges (serial)
        void find_balanced_num_edges(int nprocs, std::string file, std::vector<GraphElem>& mbins)
        {
            FILE *fp;
            GraphElem nv, ne; // #vertices, #edges
            std::vector<GraphElem> nbins(nprocs,0);
            
            fp = fopen(file.c_str(), "rb");
            if (fp == NULL) 
            {
                std::cout<< " Error opening file! " << std::endl;
                return;
            }

            // read nv and ne
            fread(&nv, sizeof(GraphElem), 1, fp);
            fread(&ne, sizeof(GraphElem), 1, fp);
          
            // bin capacity
            GraphElem nbcap = (ne / nprocs), ecount_idx, past_ecount_idx = 0;
            int p = 0;

            for (GraphElem m = 0; m < nv; m++)
            {
                fread(&ecount_idx, sizeof(GraphElem), 1, fp);
               
                // bins[p] >= capacity only for the last process
                if ((nbins[p] < nbcap) || (p == (nprocs - 1)))
                    nbins[p] += (ecount_idx - past_ecount_idx);

                // increment p as long as p is not the last process
                // worst case: excess edges piled up on (p-1)
                if ((nbins[p] >= nbcap) && (p < (nprocs - 1)))
                    p++;

                mbins[p+1]++;
                past_ecount_idx = ecount_idx;
            }
            
            fclose(fp);

            // prefix sum to store indices 
            for (int k = 1; k < nprocs+1; k++)
                mbins[k] += mbins[k-1]; 
            
            nbins.clear();
        }
        
        // read a file and return a graph
        // uses a balanced distribution
        // (approximately equal #edges per process) 
        Graph* read_balanced(int me, int nprocs, int ranks_per_node, std::string file)
        {
            int file_open_error;
            MPI_File fh;
            MPI_Status status;
            std::vector<GraphElem> mbins(nprocs+1,0);

            // find #vertices per process such that 
            // each process roughly owns equal #edges
            if (me == 0)
            {
                find_balanced_num_edges(nprocs, file, mbins);
                std::cout << "Trying to achieve equal edge distribution across processes." << std::endl;
            }
            MPI_Barrier(comm_);
            MPI_Bcast(mbins.data(), nprocs+1, MPI_GRAPH_TYPE, 0, comm_);

            // specify the number of aggregates
            MPI_Info info;
            MPI_Info_create(&info);
            int naggr = (ranks_per_node > 1) ? (nprocs/ranks_per_node) : ranks_per_node;
            if (naggr >= nprocs)
                naggr = 1;
            std::stringstream tmp_str;
            tmp_str << naggr;
            std::string str = tmp_str.str();
            MPI_Info_set(info, "cb_nodes", str.c_str());

            file_open_error = MPI_File_open(comm_, file.c_str(), MPI_MODE_RDONLY, info, &fh); 
            MPI_Info_free(&info);

            if (file_open_error != MPI_SUCCESS) 
            {
                std::cout << " Error opening file! " << std::endl;
                MPI_Abort(comm_, -99);
            }

            // read the dimensions 
            MPI_File_read_all(fh, &M_, sizeof(GraphElem), MPI_BYTE, &status);
            MPI_File_read_all(fh, &N_, sizeof(GraphElem), MPI_BYTE, &status);
            M_local_ = mbins[me+1] - mbins[me];

            // create local graph
            Graph *g = new Graph(M_local_, 0, M_, N_);
            // readjust parts with new vertex partition
            g->repart(mbins);

            uint64_t tot_bytes=(M_local_+1)*sizeof(GraphElem);
            MPI_Offset offset = 2*sizeof(GraphElem) + mbins[me]*sizeof(GraphElem);

            // read in INT_MAX increments if total byte size is > INT_MAX
            if (tot_bytes < INT_MAX)
                MPI_File_read_at(fh, offset, &g->edge_indices_[0], tot_bytes, MPI_BYTE, &status);
            else 
            {
                int chunk_bytes=INT_MAX;
                uint8_t *curr_pointer = (uint8_t*) &g->edge_indices_[0];
                uint64_t transf_bytes = 0;

                while (transf_bytes < tot_bytes)
                {
                    MPI_File_read_at(fh, offset, curr_pointer, chunk_bytes, MPI_BYTE, &status);
                    transf_bytes += chunk_bytes;
                    offset += chunk_bytes;
                    curr_pointer += chunk_bytes;

                    if ((tot_bytes - transf_bytes) < INT_MAX)
                        chunk_bytes = tot_bytes - transf_bytes;
                } 
            }    

            N_local_ = g->edge_indices_[M_local_] - g->edge_indices_[0];
            g->set_nedges(N_local_);

            tot_bytes = N_local_*(sizeof(Edge));
            offset = 2*sizeof(GraphElem) + (M_+1)*sizeof(GraphElem) + g->edge_indices_[0]*(sizeof(Edge));

            if (tot_bytes < INT_MAX)
                MPI_File_read_at(fh, offset, &g->edge_list_[0], tot_bytes, MPI_BYTE, &status);
            else 
            {
                int chunk_bytes=INT_MAX;
                uint8_t *curr_pointer = (uint8_t*)&g->edge_list_[0];
                uint64_t transf_bytes = 0;

                while (transf_bytes < tot_bytes)
                {
                    MPI_File_read_at(fh, offset, curr_pointer, chunk_bytes, MPI_BYTE, &status);
                    transf_bytes += chunk_bytes;
                    offset += chunk_bytes;
                    curr_pointer += chunk_bytes;

                    if ((tot_bytes - transf_bytes) < INT_MAX)
                        chunk_bytes = (tot_bytes - transf_bytes);
                } 
            }    

            MPI_File_close(&fh);

            for(GraphElem i=1;  i < M_local_+1; i++)
                g->edge_indices_[i] -= g->edge_indices_[0];   
            g->edge_indices_[0] = 0;

            mbins.clear();

            return g;
        }

    private:
        GraphElem M_;
        GraphElem N_;
        GraphElem M_local_;
        GraphElem N_local_;
        MPI_Comm comm_;
};

// RGG graph
// 1D vertex distribution
class GenerateRGG
{
    public:
        GenerateRGG(GraphElem nv, MPI_Comm comm = MPI_COMM_WORLD)
        {
            nv_ = nv;
            comm_ = comm;

            MPI_Comm_rank(comm_, &rank_);
            MPI_Comm_size(comm_, &nprocs_);

            // neighbors
            up_ = down_ = MPI_PROC_NULL;
            if (nprocs_ > 1) {
                if (rank_ > 0 && rank_ < (nprocs_ - 1)) {
                    up_ = rank_ - 1;
                    down_ = rank_ + 1;
                }
                if (rank_ == 0)
                    down_ = 1;
                if (rank_ == (nprocs_ - 1))
                    up_ = rank_ - 1;
            }

            n_ = nv_ / nprocs_;

            // check if number of nodes is divisible by #processes
            if ((nv_ % nprocs_) != 0) {
                if (rank_ == 0) {
                    std::cout << "[ERROR] Number of vertices must be perfectly divisible by number of processes." << std::endl;
                    std::cout << "Exiting..." << std::endl;
                }
                MPI_Abort(comm_, -99);
            }

            // check if processes are power of 2
            if (!is_pwr2(nprocs_)) {
                if (rank_ == 0) {
                    std::cout << "[ERROR] Number of processes must be a power of 2." << std::endl;
                    std::cout << "Exiting..." << std::endl;
                }
                MPI_Abort(comm_, -99);
            }

            // calculate r(n)
            GraphWeight rc = sqrt((GraphWeight)log(nv)/(GraphWeight)(PI*nv));
            GraphWeight rt = sqrt((GraphWeight)2.0736/(GraphWeight)nv);
            rn_ = (rc + rt)/(GraphWeight)2.0;
            
            assert(((GraphWeight)1.0/(GraphWeight)nprocs_) > rn_);
            
            MPI_Barrier(comm_);
        }

        // create RGG and returns Graph
        // TODO FIXME use OpenMP wherever possible
        // use Euclidean distance as edge weight
        // for random edges, choose from (0,1)
        // otherwise, use unit weight throughout
        Graph* generate(bool isLCG, bool unitEdgeWeight = true, GraphWeight randomEdgePercent = 0.0)
        {
            // Generate random coordinate points
            std::vector<GraphWeight> X, Y, X_up, Y_up, X_down, Y_down;
                       
            if (isLCG)
                X.resize(2*n_);
            else
                X.resize(n_);

            Y.resize(n_);

            if (up_ != MPI_PROC_NULL) {
                X_up.resize(n_);
                Y_up.resize(n_);
            }

            if (down_ != MPI_PROC_NULL) {
                X_down.resize(n_);
                Y_down.resize(n_);
            }
    
            // create local graph
            Graph *g = new Graph(n_, 0, nv_, nv_);

            // generate random number within range
            // X: 0, 1
            // Y: rank_*1/p, (rank_+1)*1/p,
            GraphWeight rec_np = (GraphWeight)(1.0/(GraphWeight)nprocs_);
            GraphWeight lo = rank_* rec_np; 
            GraphWeight hi = lo + rec_np;
            assert(hi > lo);

            // measure the time to generate random numbers
            MPI_Barrier(MPI_COMM_WORLD);
            double st = MPI_Wtime();

            if (!isLCG) {
                // set seed (declared an extern in utils)
                seed = (unsigned)reseeder(1);

#if defined(PRINT_RANDOM_XY_COORD)
                for (int k = 0; k < nprocs_; k++) {
                    if (k == rank_) {
                        std::cout << "Random number generated on Process#" << k << " :" << std::endl;
                        for (GraphElem i = 0; i < n_; i++) {
                            X[i] = genRandom<GraphWeight>(0.0, 1.0);
                            Y[i] = genRandom<GraphWeight>(lo, hi);
                            std::cout << "X, Y: " << X[i] << ", " << Y[i] << std::endl;
                        }
                    }
                    MPI_Barrier(comm_);
                }
#else
                for (GraphElem i = 0; i < n_; i++) {
                    X[i] = genRandom<GraphWeight>(0.0, 1.0);
                    Y[i] = genRandom<GraphWeight>(lo, hi);
                }
#endif
            }
            else { // LCG
                // X | Y
                // e.g seeds: 1741, 3821
                // create LCG object
                // seed to generate x0
                LCG xr(/*seed*/1, X.data(), 2*n_, comm_); 
                
                // generate random numbers between 0-1
                xr.generate();

                // rescale xr further between lo-hi
                // and put the numbers in Y taking
                // from X[n]
                xr.rescale(Y.data(), n_, lo);

#if defined(PRINT_RANDOM_XY_COORD)
                for (int k = 0; k < nprocs_; k++) {
                    if (k == rank_) {
                        std::cout << "Random number generated on Process#" << k << " :" << std::endl;
                        for (GraphElem i = 0; i < n_; i++) {
                            std::cout << "X, Y: " << X[i] << ", " << Y[i] << std::endl;
                        }
                    }
                    MPI_Barrier(comm_);
                }
#endif
            }
                 
            double et = MPI_Wtime();
            double tt = et - st;
            double tot_tt = 0.0;
            MPI_Reduce(&tt, &tot_tt, 1, MPI_DOUBLE, MPI_SUM, 0, comm_);
                
            if (rank_ == 0) {
                double tot_avg = (tot_tt/nprocs_);
                std::cout << "Average time to generate " << 2*n_ 
                    << " random numbers using LCG (in s): " 
                    << tot_avg << std::endl;
            }

            // ghost(s)
            
            // cross edges, each processor
            // communicates with up or/and down
            // neighbor only
            std::vector<EdgeTuple> sendup_edges, senddn_edges; 
            std::vector<EdgeTuple> recvup_edges, recvdn_edges;
            std::vector<EdgeTuple> edgeList;
            
            // counts, indexing: [2] = {up - 0, down - 1}
            // TODO can't we use MPI_INT 
            std::array<GraphElem, 2> send_sizes = {0, 0}, recv_sizes = {0, 0};
#if defined(CHECK_NUM_EDGES)
            GraphElem numEdges = 0;
#endif
            // local
            for (GraphElem i = 0; i < n_; i++) {
                for (GraphElem j = i + 1; j < n_; j++) {
                    // euclidean distance:
                    // 2D: sqrt((px-qx)^2 + (py-qy)^2)
                    GraphWeight dx = X[i] - X[j];
                    GraphWeight dy = Y[i] - Y[j];
                    GraphWeight ed = sqrt(dx*dx + dy*dy);
                    // are the two vertices within the range?
                    if (ed <= rn_) {
                        // local to global index
                        const GraphElem g_i = g->local_to_global(i);
                        const GraphElem g_j = g->local_to_global(j);

                        if (!unitEdgeWeight) {
                            edgeList.emplace_back(i, g_j, ed);
                            edgeList.emplace_back(j, g_i, ed);
                        }
                        else {
                            edgeList.emplace_back(i, g_j);
                            edgeList.emplace_back(j, g_i);
                        }
#if defined(CHECK_NUM_EDGES)
                        numEdges += 2;
#endif

                        g->edge_indices_[i+1]++;
                        g->edge_indices_[j+1]++;
                    }
                }
            }

            MPI_Barrier(comm_);
            
            // communicate ghost coordinates with neighbors
           
            const int x_ndown   = X_down.empty() ? 0 : n_;
            const int y_ndown   = Y_down.empty() ? 0 : n_;
            const int x_nup     = X_up.empty() ? 0 : n_;
            const int y_nup     = Y_up.empty() ? 0 : n_;

            MPI_Sendrecv(X.data(), n_, MPI_WEIGHT_TYPE, up_, SR_X_UP_TAG, 
                    X_down.data(), x_ndown, MPI_WEIGHT_TYPE, down_, SR_X_UP_TAG, 
                    comm_, MPI_STATUS_IGNORE);
            MPI_Sendrecv(X.data(), n_, MPI_WEIGHT_TYPE, down_, SR_X_DOWN_TAG, 
                    X_up.data(), x_nup, MPI_WEIGHT_TYPE, up_, SR_X_DOWN_TAG, 
                    comm_, MPI_STATUS_IGNORE);
            MPI_Sendrecv(Y.data(), n_, MPI_WEIGHT_TYPE, up_, SR_Y_UP_TAG, 
                    Y_down.data(), y_ndown, MPI_WEIGHT_TYPE, down_, SR_Y_UP_TAG, 
                    comm_, MPI_STATUS_IGNORE);
            MPI_Sendrecv(Y.data(), n_, MPI_WEIGHT_TYPE, down_, SR_Y_DOWN_TAG, 
                    Y_up.data(), y_nup, MPI_WEIGHT_TYPE, up_, SR_Y_DOWN_TAG, 
                    comm_, MPI_STATUS_IGNORE);
                        
            // exchange ghost vertices / cross edges
            if (nprocs_ > 1) {
                if (up_ != MPI_PROC_NULL) {
                    
                    for (GraphElem i = 0; i < n_; i++) {
                        for (GraphElem j = i + 1; j < n_; j++) {
                            GraphWeight dx = X[i] - X_up[j];
                            GraphWeight dy = Y[i] - Y_up[j];
                            GraphWeight ed = sqrt(dx*dx + dy*dy);
                            
                            if (ed <= rn_) {
                                const GraphElem g_i = g->local_to_global(i);
                                const GraphElem g_j = j + up_*n_;

                                if (!unitEdgeWeight) {
                                    sendup_edges.emplace_back(j, g_i, ed);
                                    edgeList.emplace_back(i, g_j, ed);
                                }
                                else {
                                    sendup_edges.emplace_back(j, g_i);
                                    edgeList.emplace_back(i, g_j);
                                }
#if defined(CHECK_NUM_EDGES)
                                numEdges++;
#endif
                                g->edge_indices_[i+1]++;
                            }
                        }
                    }
                    
                    // send up sizes
                    send_sizes[0] = sendup_edges.size();
                }

                if (down_ != MPI_PROC_NULL) {
                    
                    for (GraphElem i = 0; i < n_; i++) {
                        for (GraphElem j = i + 1; j < n_; j++) {
                            GraphWeight dx = X[i] - X_down[j];
                            GraphWeight dy = Y[i] - Y_down[j];
                            GraphWeight ed = sqrt(dx*dx + dy*dy);

                            if (ed <= rn_) {
                                const GraphElem g_i = g->local_to_global(i);
                                const GraphElem g_j = j + down_*n_;

                                if (!unitEdgeWeight) {
                                    senddn_edges.emplace_back(j, g_i, ed);
                                    edgeList.emplace_back(i, g_j, ed);
                                }
                                else {
                                    senddn_edges.emplace_back(j, g_i);
                                    edgeList.emplace_back(i, g_j);
                                }
#if defined(CHECK_NUM_EDGES)
                                numEdges++;
#endif
                                g->edge_indices_[i+1]++;
                            }
                        }
                    }
                    
                    // send down sizes
                    send_sizes[1] = senddn_edges.size();
                }
            }
            
            MPI_Barrier(comm_);
            
            // communicate ghost vertices with neighbors
            // send/recv buffer sizes
            
            MPI_Sendrecv(&send_sizes[0], 1, MPI_GRAPH_TYPE, up_, SR_SIZES_UP_TAG, 
                    &recv_sizes[1], 1, MPI_GRAPH_TYPE, down_, SR_SIZES_UP_TAG, 
                    comm_, MPI_STATUS_IGNORE);
            MPI_Sendrecv(&send_sizes[1], 1, MPI_GRAPH_TYPE, down_, SR_SIZES_DOWN_TAG, 
                    &recv_sizes[0], 1, MPI_GRAPH_TYPE, up_, SR_SIZES_DOWN_TAG, 
                    comm_, MPI_STATUS_IGNORE);

            // resize recv buffers
            
            if (recv_sizes[0] > 0)
                recvup_edges.resize(recv_sizes[0]);
            if (recv_sizes[1] > 0)
                recvdn_edges.resize(recv_sizes[1]);
             
            // send/recv both up and down
            
            MPI_Sendrecv(sendup_edges.data(), send_sizes[0]*sizeof(struct EdgeTuple), MPI_BYTE, 
                    up_, SR_UP_TAG, recvdn_edges.data(), recv_sizes[1]*sizeof(struct EdgeTuple), 
                    MPI_BYTE, down_, SR_UP_TAG, comm_, MPI_STATUS_IGNORE);
            MPI_Sendrecv(senddn_edges.data(), send_sizes[1]*sizeof(struct EdgeTuple), MPI_BYTE, 
                    down_, SR_DOWN_TAG, recvup_edges.data(), recv_sizes[0]*sizeof(struct EdgeTuple), 
                    MPI_BYTE, up_, SR_DOWN_TAG, comm_, MPI_STATUS_IGNORE);

            // update local #edges
            
            // down
            if (down_ != MPI_PROC_NULL) {
                for (GraphElem i = 0; i < recv_sizes[1]; i++) {
#if defined(CHECK_NUM_EDGES)
                    numEdges++;
#endif           
                    if (!unitEdgeWeight)
                        edgeList.emplace_back(recvdn_edges[i].ij_[0], recvdn_edges[i].ij_[1], recvdn_edges[i].w_);
                    else
                        edgeList.emplace_back(recvdn_edges[i].ij_[0], recvdn_edges[i].ij_[1]);
                    g->edge_indices_[recvdn_edges[i].ij_[0]+1]++; 
                } 
            }

            // up
            if (up_ != MPI_PROC_NULL) {
                for (GraphElem i = 0; i < recv_sizes[0]; i++) {
#if defined(CHECK_NUM_EDGES)
                    numEdges++;
#endif
                    if (!unitEdgeWeight)
                        edgeList.emplace_back(recvup_edges[i].ij_[0], recvup_edges[i].ij_[1], recvup_edges[i].w_);
                    else
                        edgeList.emplace_back(recvup_edges[i].ij_[0], recvup_edges[i].ij_[1]);
                    g->edge_indices_[recvup_edges[i].ij_[0]+1]++; 
                }
            }
            
            // add random edges based on 
            // randomEdgePercent 
            if (randomEdgePercent > 0.0) {
                const GraphElem pnedges = (edgeList.size()/2);
                GraphElem tot_pnedges = 0;

                MPI_Allreduce(&pnedges, &tot_pnedges, 1, MPI_GRAPH_TYPE, MPI_SUM, comm_);
                
                // extra #edges per process
                const GraphElem nrande = (((GraphElem)(randomEdgePercent * (GraphWeight)tot_pnedges))/100);
                GraphElem pnrande = 0.0;

                // TODO FIXME try to ensure a fair edge distibution
                if (nrande < nprocs_) {
                    if (rank_ == (nprocs_ - 1))
                        pnrande += nrande;
                }
                else {
                    pnrande = nrande / nprocs_;
                    const GraphElem pnrem = nrande % nprocs_;
                    if (pnrem != 0) {
                        if (rank_ == (nprocs_ - 1))
                            pnrande += pnrem;
                    }
                }
               
                // add pnrande edges 

                // send/recv buffers
                std::vector<std::vector<EdgeTuple>> rand_edges(nprocs_); 
                std::vector<EdgeTuple> sendrand_edges, recvrand_edges;

                // outgoing/incoming send/recv sizes
                // TODO FIXME if number of randomly added edges are above
                // INT_MAX, weird things will happen, fix it
                std::vector<int> sendrand_sizes(nprocs_), recvrand_sizes(nprocs_);

#if defined(PRINT_EXTRA_NEDGES)
                int extraEdges = 0;
#endif

#if defined(DEBUG_PRINTF)
                for (int i = 0; i < nprocs_; i++) {
                    if (i == rank_) {
                        std::cout << "[" << i << "]Target process for random edge insertion between " 
                            << lo << " and " << hi << std::endl;
                    }
                    MPI_Barrier(comm_);
                }
#endif
                // make sure each process has a 
                // different seed this time since
                // we want random edges
                unsigned rande_seed = (unsigned)(time(0)^getpid());
                GraphWeight weight = 1.0;
                std::hash<GraphElem> reh;
               
                // cannot use genRandom if it's already been seeded
                std::default_random_engine re(rande_seed); 
                std::uniform_int_distribution<GraphElem> IR, JR; 
                std::uniform_real_distribution<GraphWeight> IJW; 
 
                for (GraphElem k = 0; k < pnrande; k++) {

                    // randomly pick start/end vertex and target from my list
                    const GraphElem i = (GraphElem)IR(re, std::uniform_int_distribution<GraphElem>::param_type{0, (n_- 1)});
                    const GraphElem g_j = (GraphElem)JR(re, std::uniform_int_distribution<GraphElem>::param_type{0, (nv_- 1)});
                    const int target = g->get_owner(g_j);
                    const GraphElem j = g->global_to_local(g_j, target); // local

                    if (i == j) 
                        continue;

                    const GraphElem g_i = g->local_to_global(i);
                    
                    // check for duplicates prior to edgeList insertion
                    auto found = std::find_if(edgeList.begin(), edgeList.end(), 
                            [&](EdgeTuple const& et) 
                            { return ((et.ij_[0] == i) && (et.ij_[1] == g_j)); });

                    // OK to insert, not in list
                    if (found == std::end(edgeList)) { 
                   
                        // calculate weight
                        if (!unitEdgeWeight) {
                            if (target == rank_) {
                                GraphWeight dx = X[i] - X[j];
                                GraphWeight dy = Y[i] - Y[j];
                                weight = sqrt(dx*dx + dy*dy);
                            }
                            else if (target == up_) {
                                GraphWeight dx = X[i] - X_up[j];
                                GraphWeight dy = Y[i] - Y_up[j];
                                weight = sqrt(dx*dx + dy*dy);
                            }
                            else if (target == down_) {
                                GraphWeight dx = X[i] - X_down[j];
                                GraphWeight dy = Y[i] - Y_down[j];
                                weight = sqrt(dx*dx + dy*dy);
                            }
                            else {
                                unsigned randw_seed = reh((GraphElem)(g_i*nv_+g_j));
                                std::default_random_engine rew(randw_seed); 
                                weight = (GraphWeight)IJW(rew, std::uniform_real_distribution<GraphWeight>::param_type{0.01, 1.0});
                            }
                        }

                        rand_edges[target].emplace_back(j, g_i, weight);
                        sendrand_sizes[target]++;

#if defined(PRINT_EXTRA_NEDGES)
                        extraEdges++;
#endif
#if defined(CHECK_NUM_EDGES)
                        numEdges++;
#endif                       
                        edgeList.emplace_back(i, g_j, weight);
                        g->edge_indices_[i+1]++;
                    }
                }
                
#if defined(PRINT_EXTRA_NEDGES)
                int totExtraEdges = 0;
                MPI_Reduce(&extraEdges, &totExtraEdges, 1, MPI_INT, MPI_SUM, 0, comm_);
                if (rank_ == 0)
                    std::cout << "Adding extra " << totExtraEdges << " edges while trying to incorporate " 
                        << randomEdgePercent << "%" << " extra edges globally." << std::endl;
#endif

                MPI_Barrier(comm_);
              
                // communicate ghosts edges
                MPI_Request rande_sreq;

                MPI_Ialltoall(sendrand_sizes.data(), 1, MPI_INT, 
                        recvrand_sizes.data(), 1, MPI_INT, comm_, 
                        &rande_sreq);

                // send data if outgoing size > 0
                for (int p = 0; p < nprocs_; p++) {
                    sendrand_edges.insert(sendrand_edges.end(), 
                            rand_edges[p].begin(), rand_edges[p].end());
                }

                MPI_Wait(&rande_sreq, MPI_STATUS_IGNORE);
               
                // total recvbuffer size
                const int rcount = std::accumulate(recvrand_sizes.begin(), recvrand_sizes.end(), 0);
                recvrand_edges.resize(rcount);
                                
                // alltoallv for incoming data
                // TODO FIXME make sure size of extra edges is 
                // within INT limits
               
                int rpos = 0, spos = 0;
                std::vector<int> sdispls(nprocs_), rdispls(nprocs_);
                
                for (int p = 0; p < nprocs_; p++) {

                    sendrand_sizes[p] *= sizeof(struct EdgeTuple);
                    recvrand_sizes[p] *= sizeof(struct EdgeTuple);
                    
                    sdispls[p] = spos;
                    rdispls[p] = rpos;
                    
                    spos += sendrand_sizes[p];
                    rpos += recvrand_sizes[p];
                }
                
                MPI_Alltoallv(sendrand_edges.data(), sendrand_sizes.data(), sdispls.data(), 
                        MPI_BYTE, recvrand_edges.data(), recvrand_sizes.data(), rdispls.data(), 
                        MPI_BYTE, comm_);
                
                // update local edge list
                for (int i = 0; i < rcount; i++) {
#if defined(CHECK_NUM_EDGES)
                    numEdges++;
#endif
                    edgeList.emplace_back(recvrand_edges[i].ij_[0], recvrand_edges[i].ij_[1], recvrand_edges[i].w_);
                    g->edge_indices_[recvrand_edges[i].ij_[0]+1]++; 
                }

                sendrand_edges.clear();
                recvrand_edges.clear();
                rand_edges.clear();
            } // end of (conditional) random edges addition

            MPI_Barrier(comm_);
  
            // set graph edge indices
            
            std::vector<GraphElem> ecTmp(n_+1);
            std::partial_sum(g->edge_indices_.begin(), g->edge_indices_.end(), ecTmp.begin());
            g->edge_indices_ = ecTmp;
             
            for(GraphElem i = 1; i < n_+1; i++)
                g->edge_indices_[i] -= g->edge_indices_[0];   
            g->edge_indices_[0] = 0;

            g->set_edge_index(0, 0);
            for (GraphElem i = 0; i < n_; i++)
                g->set_edge_index(i+1, g->edge_indices_[i+1]);
            
            const GraphElem nedges = g->edge_indices_[n_] - g->edge_indices_[0];
            g->set_nedges(nedges);
            
            // set graph edge list
            // sort edge list
            auto ecmp = [] (EdgeTuple const& e0, EdgeTuple const& e1)
            { return ((e0.ij_[0] < e1.ij_[0]) || ((e0.ij_[0] == e1.ij_[0]) && (e0.ij_[1] < e1.ij_[1]))); };

            if (!std::is_sorted(edgeList.begin(), edgeList.end(), ecmp)) {
#if defined(DEBUG_PRINTF)
                std::cout << "Edge list is not sorted." << std::endl;
#endif
                std::sort(edgeList.begin(), edgeList.end(), ecmp);
            }
#if defined(DEBUG_PRINTF)
            else
                std::cout << "Edge list is sorted!" << std::endl;
#endif
  
            GraphElem ePos = 0;
            for (GraphElem i = 0; i < n_; i++) {
                GraphElem e0, e1;

                g->edge_range(i, e0, e1);
#if defined(DEBUG_PRINTF)
                if ((i % 100000) == 0)
                    std::cout << "Processing edges for vertex: " << i << ", range(" << e0 << ", " << e1 <<
                        ")" << std::endl;
#endif
                for (GraphElem j = e0; j < e1; j++) {
                    Edge &edge = g->set_edge(j);

                    assert(ePos == j);
                    assert(i == edgeList[ePos].ij_[0]);
                    
                    edge.tail_ = edgeList[ePos].ij_[1];
                    edge.weight_ = edgeList[ePos].w_;

                    ePos++;
                }
            }
            
#if defined(CHECK_NUM_EDGES)
            GraphElem tot_numEdges = 0;
            MPI_Allreduce(&numEdges, &tot_numEdges, 1, MPI_GRAPH_TYPE, MPI_SUM, comm_);
            const GraphElem tne = g->get_ne();
            assert(tne == tot_numEdges);
#endif
            edgeList.clear();
            
            X.clear();
            Y.clear();
            X_up.clear();
            Y_up.clear();
            X_down.clear();
            Y_down.clear();

            sendup_edges.clear();
            senddn_edges.clear();
            recvup_edges.clear();
            recvdn_edges.clear();

            return g;
        }

        GraphWeight get_d() const { return rn_; }
        GraphElem get_nv() const { return nv_; }

    private:
        GraphElem nv_, n_;
        GraphWeight rn_;
        MPI_Comm comm_;
        int nprocs_, rank_, up_, down_;
};

#endif
