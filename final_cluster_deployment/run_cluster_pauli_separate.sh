#!/bin/bash
#SBATCH -p batch
#SBATCH --job-name hpc_gap_pauli_separate
#SBATCH -N 1 # nodes requested
#SBATCH -n 1 # tasks requested
#SBATCH -c 8 # cores requested (for multiprocessing)
#SBATCH -t 12:00:00 # 12 hours time limit
#SBATCH --mem=64000 # memory in Mb (64GB) - for Pauli Hamiltonians
#SBATCH -o hpc_gap_pauli_separate_output.out # send stdout to outfile
#SBATCH -e hpc_gap_pauli_separate_error.out  # send stderr to errfile

# Load necessary modules (NO OpenFermion needed!)
module load miniforge/24.11.2-py312

# Activate conda environment (should have numpy, scipy, matplotlib, symmer)
source activate /cluster/tufts/lovelab/fqian03/condaenv/gap

echo "Starting HPC Gap Analysis (PAULI VERSION - SEPARATE MOLECULES)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Cores: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Started at: $(date)"

# Set Python path to include current directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check for Symmer availability
python -c "import symmer; print(f'✓ Symmer version: {symmer.__version__}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "WARNING: Symmer not available, using fallback Pauli implementation"
else
    echo "✓ Symmer library available"
fi

# Find all available Pauli Hamiltonian files
echo "Auto-detecting available molecules..."
MOLECULES=($(ls *_pauli_hamiltonian.json 2>/dev/null | sed 's/_pauli_hamiltonian.json//g'))

if [ ${#MOLECULES[@]} -eq 0 ]; then
    echo "ERROR: No Pauli Hamiltonian files found!"
    echo "Please run 'python generate_pauli_hamiltonians.py' locally first"
    echo "and transfer the *_pauli_hamiltonian.json files to the cluster"
    exit 1
fi

echo "Found ${#MOLECULES[@]} molecules: ${MOLECULES[@]}"

# Check if this is a test run
if [ "$1" = "test" ]; then
    echo "Running in TEST mode with first molecule only"
    MOLECULES=(${MOLECULES[0]})
fi

# Create results directory
mkdir -p individual_results

# Process each molecule separately
SUCCESSFUL_MOLECULES=()
FAILED_MOLECULES=()

for MOLECULE in "${MOLECULES[@]}"; do
    echo ""
    echo "=" $(printf "%0.s=" {1..50})
    echo "PROCESSING MOLECULE: $MOLECULE"
    echo "=" $(printf "%0.s=" {1..50})
    echo "Started at: $(date)"

    # Create individual output files
    OUTPUT_FILE="individual_results/${MOLECULE}_output.out"
    ERROR_FILE="individual_results/${MOLECULE}_error.out"
    RESULT_FILE="individual_results/${MOLECULE}_results.json"

    # Run analysis for single molecule
    echo "Running analysis for $MOLECULE..."
    python main_cluster_pauli.py --molecules $MOLECULE --processes 1 > "$OUTPUT_FILE" 2> "$ERROR_FILE"

    # Check exit status
    if [ $? -eq 0 ]; then
        echo "✓ $MOLECULE completed successfully at: $(date)"
        SUCCESSFUL_MOLECULES+=($MOLECULE)

        # Move results file if it exists
        if [ -f "pauli_analysis_results.json" ]; then
            mv "pauli_analysis_results.json" "$RESULT_FILE"
            echo "  Results saved to: $RESULT_FILE"
        fi

        # Show summary from output
        echo "  Summary from $MOLECULE:"
        tail -n 5 "$OUTPUT_FILE" | grep -E "(Total time|Results saved|COMPLETE)"

    else
        echo "✗ $MOLECULE failed at: $(date)"
        FAILED_MOLECULES+=($MOLECULE)

        # Show error summary
        echo "  Error summary for $MOLECULE:"
        tail -n 10 "$ERROR_FILE"
    fi

    echo "Finished $MOLECULE at: $(date)"
done

# Final summary
echo ""
echo "=" $(printf "%0.s=" {1..60})
echo "FINAL SUMMARY"
echo "=" $(printf "%0.s=" {1..60})
echo "Total molecules processed: ${#MOLECULES[@]}"
echo "Successful: ${#SUCCESSFUL_MOLECULES[@]} - ${SUCCESSFUL_MOLECULES[@]}"
echo "Failed: ${#FAILED_MOLECULES[@]} - ${FAILED_MOLECULES[@]}"

# Combine all successful results into master file
if [ ${#SUCCESSFUL_MOLECULES[@]} -gt 0 ]; then
    echo ""
    echo "Combining results from successful molecules..."

    # Create master results file
    echo "{" > master_pauli_results.json
    echo "  \"analysis_complete\": true," >> master_pauli_results.json
    echo "  \"total_molecules\": ${#MOLECULES[@]}," >> master_pauli_results.json
    echo "  \"successful_molecules\": ${#SUCCESSFUL_MOLECULES[@]}," >> master_pauli_results.json
    echo "  \"failed_molecules\": ${#FAILED_MOLECULES[@]}," >> master_pauli_results.json
    echo "  \"successful_list\": [$(printf '"%s",' "${SUCCESSFUL_MOLECULES[@]}" | sed 's/,$//')]," >> master_pauli_results.json
    echo "  \"failed_list\": [$(printf '"%s",' "${FAILED_MOLECULES[@]}" | sed 's/,$//')]," >> master_pauli_results.json
    echo "  \"individual_results\": {" >> master_pauli_results.json

    FIRST=true
    for MOLECULE in "${SUCCESSFUL_MOLECULES[@]}"; do
        if [ "$FIRST" = true ]; then
            FIRST=false
        else
            echo "," >> master_pauli_results.json
        fi
        echo -n "    \"$MOLECULE\": " >> master_pauli_results.json
        if [ -f "individual_results/${MOLECULE}_results.json" ]; then
            cat "individual_results/${MOLECULE}_results.json" >> master_pauli_results.json
        else
            echo "null" >> master_pauli_results.json
        fi
    done

    echo "" >> master_pauli_results.json
    echo "  }" >> master_pauli_results.json
    echo "}" >> master_pauli_results.json

    echo "✓ Master results saved to: master_pauli_results.json"
fi

# Create completion timestamp
echo "Analysis completed at: $(date)" > completion_timestamp_separate.txt
echo "Successful molecules: ${SUCCESSFUL_MOLECULES[@]}" >> completion_timestamp_separate.txt
echo "Failed molecules: ${FAILED_MOLECULES[@]}" >> completion_timestamp_separate.txt

echo ""
echo "Job finished at: $(date)"

# Exit with error if any molecules failed
if [ ${#FAILED_MOLECULES[@]} -gt 0 ]; then
    echo "Exiting with error due to failed molecules"
    exit 1
else
    echo "All molecules completed successfully!"
    exit 0
fi
