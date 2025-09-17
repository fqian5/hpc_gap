#!/bin/bash
# Submit separate SLURM jobs for each molecule to avoid interference

echo "SUBMITTING SEPARATE JOBS FOR ALL MOLECULES"
echo "=========================================="

# Find all available Pauli Hamiltonian files
MOLECULES=($(ls *_pauli_hamiltonian.json 2>/dev/null | sed 's/_pauli_hamiltonian.json//g'))

if [ ${#MOLECULES[@]} -eq 0 ]; then
    echo "ERROR: No Pauli Hamiltonian files found!"
    echo "Please ensure *_pauli_hamiltonian.json files are in the current directory"
    exit 1
fi

echo "Found ${#MOLECULES[@]} molecules: ${MOLECULES[@]}"
echo ""

# Create individual job scripts and submit each molecule
JOB_IDS=()

for MOLECULE in "${MOLECULES[@]}"; do
    JOB_SCRIPT="run_${MOLECULE}.sh"

    echo "Creating job script for $MOLECULE: $JOB_SCRIPT"

    # Create individual SLURM job script
    cat > "$JOB_SCRIPT" << EOF
#!/bin/bash
#SBATCH -p batch
#SBATCH --job-name gap_${MOLECULE}
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -t 8:00:00
#SBATCH --mem=32000
#SBATCH -o ${MOLECULE}_output.out
#SBATCH -e ${MOLECULE}_error.out

# Load necessary modules
module load miniforge/24.11.2-py312

# Activate conda environment
source activate /cluster/tufts/lovelab/fqian03/condaenv/gap

echo "Starting HPC Gap Analysis for $MOLECULE"
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURMD_NODENAME"
echo "Started at: \$(date)"

# Set Python path
export PYTHONPATH="\${PYTHONPATH}:\$(pwd)"

# Check Symmer availability
python -c "import symmer; print(f'✓ Symmer version: {symmer.__version__}')" 2>/dev/null
if [ \$? -ne 0 ]; then
    echo "WARNING: Symmer not available, using fallback"
else
    echo "✓ Symmer library available"
fi

echo "Processing molecule: $MOLECULE"
echo "=================================="

# Create molecule-specific results directory
mkdir -p ${MOLECULE}_results

# Run analysis for this specific molecule
python main_cluster_pauli.py --molecules $MOLECULE --processes 1

# Check exit status
if [ \$? -eq 0 ]; then
    echo "✓ $MOLECULE completed successfully at: \$(date)"

    # Move results to molecule-specific directory
    if [ -f "pauli_analysis_results.json" ]; then
        mv "pauli_analysis_results.json" "${MOLECULE}_results/${MOLECULE}_results.json"
        echo "Results saved to: ${MOLECULE}_results/${MOLECULE}_results.json"
    fi

    # Move any generated plots
    if [ -d "${MOLECULE}_plots" ]; then
        mv "${MOLECULE}_plots" "${MOLECULE}_results/"
        echo "Plots moved to: ${MOLECULE}_results/${MOLECULE}_plots/"
    fi

    # Create success marker
    echo "SUCCESS: $MOLECULE analysis completed at \$(date)" > "${MOLECULE}_results/SUCCESS"

else
    echo "✗ $MOLECULE failed at: \$(date)"
    echo "FAILED: $MOLECULE analysis failed at \$(date)" > "${MOLECULE}_results/FAILED"
    exit 1
fi

echo "Job finished at: \$(date)"
EOF

    # Make script executable
    chmod +x "$JOB_SCRIPT"

    # Submit the job
    JOB_OUTPUT=$(sbatch "$JOB_SCRIPT")
    JOB_ID=$(echo "$JOB_OUTPUT" | grep -o '[0-9]*')
    JOB_IDS+=($JOB_ID)

    echo "  ✓ Submitted job $JOB_ID for $MOLECULE"

done

echo ""
echo "=========================================="
echo "ALL JOBS SUBMITTED"
echo "=========================================="
echo "Total jobs: ${#JOB_IDS[@]}"
echo "Job IDs: ${JOB_IDS[@]}"
echo ""

# Create monitoring script
cat > "monitor_jobs.sh" << EOF
#!/bin/bash
echo "Monitoring submitted jobs..."
echo "Job IDs: ${JOB_IDS[@]}"
echo ""

# Check job status
squeue -j $(IFS=,; echo "${JOB_IDS[*]}")

echo ""
echo "To check individual job status:"
for i in "\${!JOB_IDS[@]}"; do
    echo "  squeue -j \${JOB_IDS[\$i]}  # \${MOLECULES[\$i]}"
done

echo ""
echo "To check completed results:"
for MOLECULE in ${MOLECULES[@]}; do
    echo "  ls -la \${MOLECULE}_results/"
done

echo ""
echo "To collect all results when jobs complete:"
echo "  bash collect_results.sh"
EOF

# Create results collection script
cat > "collect_results.sh" << EOF
#!/bin/bash
echo "Collecting results from all molecules..."
echo ""

SUCCESSFUL=()
FAILED=()

# Check each molecule's results
for MOLECULE in ${MOLECULES[@]}; do
    if [ -f "\${MOLECULE}_results/SUCCESS" ]; then
        SUCCESSFUL+=(\$MOLECULE)
        echo "✓ \$MOLECULE: SUCCESS"
    elif [ -f "\${MOLECULE}_results/FAILED" ]; then
        FAILED+=(\$MOLECULE)
        echo "✗ \$MOLECULE: FAILED"
    else
        echo "? \$MOLECULE: UNKNOWN (still running or not started)"
    fi
done

echo ""
echo "SUMMARY:"
echo "Successful: \${#SUCCESSFUL[@]} - \${SUCCESSFUL[@]}"
echo "Failed: \${#FAILED[@]} - \${FAILED[@]}"

if [ \${#SUCCESSFUL[@]} -gt 0 ]; then
    echo ""
    echo "Creating combined results file..."

    echo "{" > combined_results.json
    echo "  \\"total_molecules\\": ${#MOLECULES[@]}," >> combined_results.json
    echo "  \\"successful\\": \${#SUCCESSFUL[@]}," >> combined_results.json
    echo "  \\"failed\\": \${#FAILED[@]}," >> combined_results.json
    echo "  \\"results\\": {" >> combined_results.json

    FIRST=true
    for MOLECULE in "\${SUCCESSFUL[@]}"; do
        if [ "\$FIRST" = true ]; then
            FIRST=false
        else
            echo "," >> combined_results.json
        fi
        echo -n "    \\"\$MOLECULE\\": " >> combined_results.json
        if [ -f "\${MOLECULE}_results/\${MOLECULE}_results.json" ]; then
            cat "\${MOLECULE}_results/\${MOLECULE}_results.json" >> combined_results.json
        else
            echo "null" >> combined_results.json
        fi
    done

    echo "" >> combined_results.json
    echo "  }" >> combined_results.json
    echo "}" >> combined_results.json

    echo "✓ Combined results saved to: combined_results.json"
fi
EOF

chmod +x monitor_jobs.sh
chmod +x collect_results.sh

echo "Helper scripts created:"
echo "  monitor_jobs.sh  - Check job status"
echo "  collect_results.sh - Collect results when complete"
echo ""
echo "Usage:"
echo "  bash monitor_jobs.sh     # Check job status"
echo "  bash collect_results.sh  # Collect results after jobs finish"
echo ""
echo "Individual job outputs will be in:"
for MOLECULE in "${MOLECULES[@]}"; do
    echo "  ${MOLECULE}_output.out, ${MOLECULE}_error.out, ${MOLECULE}_results/"
done