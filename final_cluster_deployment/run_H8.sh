#!/bin/bash
#SBATCH -p batch
#SBATCH --job-name gap_H8
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 8:00:00
#SBATCH --mem=32000
#SBATCH -o H8_output.out
#SBATCH -e H8_error.out

# Load necessary modules
module load miniforge/24.11.2-py312

# Activate conda environment
source activate /cluster/tufts/lovelab/fqian03/condaenv/gap

echo "Starting HPC Gap Analysis for H8"
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

echo "Processing molecule: H8"
echo "=================================="

# Create molecule-specific results directory
mkdir -p H8_results

# Run analysis for this specific molecule
python main_cluster_pauli.py --molecules H8 --processes 1

# Check exit status
if [ $? -eq 0 ]; then
    echo "✓ H8 completed successfully at: $(date)"

    # Move results to molecule-specific directory
    if [ -f "pauli_analysis_results.json" ]; then
        mv "pauli_analysis_results.json" "H8_results/H8_results.json"
        echo "Results saved to: H8_results/H8_results.json"
    fi

    # Move any generated plots
    if [ -d "H8_plots" ]; then
        mv "H8_plots" "H8_results/"
        echo "Plots moved to: H8_results/H8_plots/"
    fi

    # Create success marker
    echo "SUCCESS: H8 analysis completed at $(date)" > "H8_results/SUCCESS"

else
    echo "✗ H8 failed at: $(date)"
    echo "FAILED: H8 analysis failed at $(date)" > "H8_results/FAILED"
    exit 1
fi

echo "Job finished at: $(date)"
