#!/bin/bash
echo "Monitoring submitted jobs..."
echo "Job IDs: 15700804 15700805 15700806 15700807 15700808 15700809"
echo ""

# Check job status
squeue -j 15700804,15700805,15700806,15700807,15700808,15700809

echo ""
echo "To check individual job status:"
for i in "${!JOB_IDS[@]}"; do
    echo "  squeue -j ${JOB_IDS[$i]}  # ${MOLECULES[$i]}"
done

echo ""
echo "To check completed results:"
for MOLECULE in H2O H2 H4 H6 H8 N2; do
    echo "  ls -la ${MOLECULE}_results/"
done

echo ""
echo "To collect all results when jobs complete:"
echo "  bash collect_results.sh"
