#!/bin/sh
#SBATCH --partition=slurm_shortgpu
#SBATCH --time=0-00:20:00 # run time in days-hh:mm:ss
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --error=OMP1_verify.err
#SBATCH --output=OMP1_verify.out

uname -a
./neuralnetOMP1_verify 5 25 0.1 5

