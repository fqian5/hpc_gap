#!/bin/bash
#SBATCH -p batch
#SBATCH --job-name h7
#SBATCH -N 1 # nodes requested
#SBATCH -n 1 # tasks requested
#SBATCH -c 5 # cores requested
#SBATCH -t 3-00:00:00
#SBATCH --mem=500000 # memory in Mb
#SBATCH -o outputfile7.out # send stdout to outfile
#SBATCH -e errfile7.out  # send stderr to errfile
module load miniforge/24.11.2-py312
source activate /cluster/tufts/lovelab/fqian03/condaenv/gap
python h7.py 1>out7 2>error7
