#!/bin/bash
#SBATCH -p batch
#SBATCH --job-name gap_N2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 8:00:00
#SBATCH --mem=32000
#SBATCH -o N2_output.out
#SBATCH -e N2_error.out

# Load necessary modules
module load miniforge/24.11.2-py312

# Activate conda environment
source activate /cluster/tufts/lovelab/fqian03/condaenv/gap

echo "Starting HPC Gap Analysis for N2"
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

echo "Processing molecule: N2"
echo "=================================="

# Create molecule-specific results directory
mkdir -p N2_results

# Run analysis for this specific molecule
python main_cluster_pauli.py --molecules N2 --processes 1

# Check exit status
if [ $? -eq 0 ]; then
    echo "✓ N2 completed successfully at: $(date)"

    # Move results to molecule-specific directory
    if [ -f "pauli_analysis_results.json" ]; then
        mv "pauli_analysis_results.json" "N2_results/N2_results.json"
        echo "Results saved to: N2_results/N2_results.json"
    fi

    # Move any generated plots
    if [ -d "N2_plots" ]; then
        mv "N2_plots" "N2_results/"
        echo "Plots moved to: N2_results/N2_plots/"
    fi

    # Create success marker
    echo "SUCCESS: N2 analysis completed at $(date)" > "N2_results/SUCCESS"

else
    echo "✗ N2 failed at: $(date)"
    echo "FAILED: N2 analysis failed at $(date)" > "N2_results/FAILED"
    exit 1
fi

echo "Job finished at: $(date)"
