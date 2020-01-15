#!/bin/bash
#SBATCH --time=48:00:00   # walltime
#SBATCH --ntasks=12   # number of processor cores (i.e. tasks)
#SBATCH -C 'intel'   # features syntax (use quotes): -C 'feature_name'
#SBATCH --mem-per-cpu=100M   # memory per CPU core
#SBATCH -J "test"   # job name
#SBATCH --output=result_mpi.txt

module load mpi/openmpi-x86_64
mpirun --map-by core --mca btl self,vader,tcp ./program 
