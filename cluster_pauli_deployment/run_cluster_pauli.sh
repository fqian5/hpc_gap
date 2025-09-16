#!/bin/bash
#SBATCH -p batch
#SBATCH --job-name hpc_gap_pauli
#SBATCH -N 1 # nodes requested
#SBATCH -n 1 # tasks requested
#SBATCH -c 8 # cores requested (for multiprocessing)
#SBATCH -t 12:00:00 # 12 hours time limit
#SBATCH --mem=64000 # memory in Mb (64GB) - for Pauli Hamiltonians
#SBATCH -o hpc_gap_pauli_output.out # send stdout to outfile
#SBATCH -e hpc_gap_pauli_error.out  # send stderr to errfile

# Load necessary modules (NO OpenFermion needed!)
module load miniforge/24.11.2-py312

# Activate conda environment (should have numpy, scipy, matplotlib, symmer)
source activate /cluster/tufts/lovelab/fqian03/condaenv/iadiabatic

echo "Starting HPC Gap Analysis (PAULI VERSION - SYMMER)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Cores: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Started at: $(date)"

# Set Python path to include current directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check if Pauli Hamiltonian files exist
echo "Checking for pre-generated Pauli Hamiltonian files..."
if [ ! -f "H2_pauli_hamiltonian.json" ]; then
    echo "ERROR: H2_pauli_hamiltonian.json not found!"
    echo "Please run 'python generate_pauli_hamiltonians.py' locally first"
    echo "and transfer the *_pauli_hamiltonian.json files to the cluster"
    exit 1
fi

echo "✓ Pre-generated Pauli Hamiltonian files found"

# Check for Symmer availability
python -c "import symmer; print(f'✓ Symmer version: {symmer.__version__}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "WARNING: Symmer not available, using fallback Pauli implementation"
else
    echo "✓ Symmer library available"
fi

# Check if this is a test run or full run
if [ "$1" = "test" ]; then
    echo "Running in TEST mode with H2 molecule only"
    python main_cluster_pauli.py --test --processes $SLURM_CPUS_PER_TASK
else
    echo "Running FULL analysis with all molecules"
    python main_cluster_pauli.py --processes $SLURM_CPUS_PER_TASK
fi

# Check exit status
if [ $? -eq 0 ]; then
    echo "Pauli cluster analysis completed successfully at: $(date)"
else
    echo "Pauli cluster analysis failed at: $(date)"
    exit 1
fi

# Create a completion timestamp file
echo "Completed at: $(date)" > completion_timestamp_pauli.txt

echo "Job finished at: $(date)"