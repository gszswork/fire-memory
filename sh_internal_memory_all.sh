#!/bin/bash
# Run FIRE internal memory (verify_atomic_claim with web search) on all datasets

DATASETS=("averitec" "aggrefact_cnn" "pubhealth" "summeval" "LIAR-New" "hover")

for dataset in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "Evaluating: $dataset"
    echo "=========================================="
    python run_fire_internal_memory.py --benchmark "$dataset"
    echo ""
done

echo "=========================================="
echo "All evaluations completed!"
echo "Results saved in: results/fire_internal_memory_*/"
echo "=========================================="
