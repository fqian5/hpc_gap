#!/bin/bash
echo "Collecting results from all molecules..."
echo ""

SUCCESSFUL=()
FAILED=()

# Check each molecule's results
for MOLECULE in H2O H2 H4 H6 H8 N2; do
    if [ -f "${MOLECULE}_results/SUCCESS" ]; then
        SUCCESSFUL+=($MOLECULE)
        echo "✓ $MOLECULE: SUCCESS"
    elif [ -f "${MOLECULE}_results/FAILED" ]; then
        FAILED+=($MOLECULE)
        echo "✗ $MOLECULE: FAILED"
    else
        echo "? $MOLECULE: UNKNOWN (still running or not started)"
    fi
done

echo ""
echo "SUMMARY:"
echo "Successful: ${#SUCCESSFUL[@]} - ${SUCCESSFUL[@]}"
echo "Failed: ${#FAILED[@]} - ${FAILED[@]}"

if [ ${#SUCCESSFUL[@]} -gt 0 ]; then
    echo ""
    echo "Creating combined results file..."

    echo "{" > combined_results.json
    echo "  \"total_molecules\": 6," >> combined_results.json
    echo "  \"successful\": ${#SUCCESSFUL[@]}," >> combined_results.json
    echo "  \"failed\": ${#FAILED[@]}," >> combined_results.json
    echo "  \"results\": {" >> combined_results.json

    FIRST=true
    for MOLECULE in "${SUCCESSFUL[@]}"; do
        if [ "$FIRST" = true ]; then
            FIRST=false
        else
            echo "," >> combined_results.json
        fi
        echo -n "    \"$MOLECULE\": " >> combined_results.json
        if [ -f "${MOLECULE}_results/${MOLECULE}_results.json" ]; then
            cat "${MOLECULE}_results/${MOLECULE}_results.json" >> combined_results.json
        else
            echo "null" >> combined_results.json
        fi
    done

    echo "" >> combined_results.json
    echo "  }" >> combined_results.json
    echo "}" >> combined_results.json

    echo "✓ Combined results saved to: combined_results.json"
fi
