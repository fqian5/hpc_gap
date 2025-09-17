#!/bin/bash
#SBATCH -p batch
#SBATCH --job-name gap_H6
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 8:00:00
#SBATCH --mem=32000
#SBATCH -o H6_output.out
#SBATCH -e H6_error.out

# Load necessary modules
module load miniforge/24.11.2-py312

# Activate conda environment
source activate /cluster/tufts/lovelab/fqian03/condaenv/gap

echo "Starting HPC Gap Analysis for H6"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started at: $(date)"

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check Symmer availability
python -c "import symmer; print(f'✓ Symmer version: {symmer.__version__}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "WARNING: Symmer not available, using fallback"
else
    echo "✓ Symmer library available"
fi

echo "Processing molecule: H6"
echo "=================================="

# Create molecule-specific results directory
mkdir -p H6_results

# Run analysis for this specific molecule
python main_cluster_pauli.py --molecules H6 --processes 1

# Check exit status
if [ $? -eq 0 ]; then
    echo "✓ H6 completed successfully at: $(date)"

    # Move results to molecule-specific directory
    if [ -f "pauli_analysis_results.json" ]; then
        mv "pauli_analysis_results.json" "H6_results/H6_results.json"
        echo "Results saved to: H6_results/H6_results.json"
    fi

    # Move any generated plots
    if [ -d "H6_plots" ]; then
        mv "H6_plots" "H6_results/"
        echo "Plots moved to: H6_results/H6_plots/"
    fi

    # Create success marker
    echo "SUCCESS: H6 analysis completed at $(date)" > "H6_results/SUCCESS"

else
    echo "✗ H6 failed at: $(date)"
    echo "FAILED: H6 analysis failed at $(date)" > "H6_results/FAILED"
    exit 1
fi

echo "Job finished at: $(date)"
