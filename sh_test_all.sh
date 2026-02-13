#!/bin/bash
# Test RAG-only evaluation on all datasets with all ratios
# No web retrieval - only RAG knowledge is used
#
# Ratios:
#   0   - No RAG evidence (baseline, tests pure model reasoning)
#   0.2 - 20% of training evidence visible
#   0.4 - 40% of training evidence visible
#   0.6 - 60% of training evidence visible
#   0.8 - 80% of training evidence visible (max available)

# Datasets
# DATASETS=("averitec" "aggrefact_cnn" "pubhealth" "summeval" "LIAR-New")
DATASETS=("averitec")

# Ratios to test
RATIOS=(0)

# Run tests
for dataset in "${DATASETS[@]}"; do
    for ratio in "${RATIOS[@]}"; do
        echo "=========================================="
        echo "Testing: $dataset with train_ratio=$ratio"
        echo "=========================================="
        python run_fire_rag_test.py --benchmark "$dataset" --build_ratio 0.8 --train_ratio "$ratio"
        echo ""
    done
done

echo "=========================================="
echo "All tests completed!"
echo "Results saved in: results_test/"
echo "=========================================="
