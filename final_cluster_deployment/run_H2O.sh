#!/bin/bash
#SBATCH -p batch
#SBATCH --job-name gap_H2O
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 8:00:00
#SBATCH --mem=32000
#SBATCH -o H2O_output.out
#SBATCH -e H2O_error.out

# Load necessary modules
module load miniforge/24.11.2-py312

# Activate conda environment
source activate /cluster/tufts/lovelab/fqian03/condaenv/gap

echo "Starting HPC Gap Analysis for H2O"
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

echo "Processing molecule: H2O"
echo "=================================="

# Create molecule-specific results directory
mkdir -p H2O_results

# Run analysis for this specific molecule
python main_cluster_pauli.py --molecules H2O --processes 1

# Check exit status
if [ $? -eq 0 ]; then
    echo "✓ H2O completed successfully at: $(date)"

    # Move results to molecule-specific directory
    if [ -f "pauli_analysis_results.json" ]; then
        mv "pauli_analysis_results.json" "H2O_results/H2O_results.json"
        echo "Results saved to: H2O_results/H2O_results.json"
    fi

    # Move any generated plots
    if [ -d "H2O_plots" ]; then
        mv "H2O_plots" "H2O_results/"
        echo "Plots moved to: H2O_results/H2O_plots/"
    fi

    # Create success marker
    echo "SUCCESS: H2O analysis completed at $(date)" > "H2O_results/SUCCESS"

else
    echo "✗ H2O failed at: $(date)"
    echo "FAILED: H2O analysis failed at $(date)" > "H2O_results/FAILED"
    exit 1
fi

echo "Job finished at: $(date)"
