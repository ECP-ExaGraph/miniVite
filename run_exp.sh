#!/bin/bash

export BINPATH=$HOME/miniVite_folk/build
export METPATH=/l/ssd/graph
export OMP_NUM_THREADS=2

echo "miniVite-metall STORE and LOAD"
export n_vs=1024
# export n_vs=1048576

# for t in 8 16 32 64
for t in 2
do
  export procs=$((t*2))
  export n_v=$(($n_vs*$t))
  export BIN_ARGS_STORE="-l -w -n $n_v -s $METPATH"
  export BIN_ARGS_LOAD="-c $METPATH"
  echo "STORE on $procs"
  srun -n ${procs} --clear-ssd $BINPATH/./miniVite $BIN_ARGS_STORE
  echo "LOAD on $procs"
  srun -n ${procs} --drop-caches=pagecache $BINPATH/./miniVite $BIN_ARGS_LOAD
done