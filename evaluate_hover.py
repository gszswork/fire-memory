#!/usr/bin/env python3
"""
Evaluate fact-verification results for fire_hover experiments.
Calculates accuracy and balanced accuracy.

Ground truth: SUPPORTED / NOT_SUPPORTED
Predicted: True / False

Mapping: SUPPORTED -> True, NOT_SUPPORTED -> False
"""

import json
from pathlib import Path
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report


def load_results(filepath: str) -> list[dict]:
    """Load results from a JSONL file."""
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def calculate_hit_rate(results: list[dict]) -> float | None:
    """
    Calculate memory hit_rate@1 if the data is available.

    Returns:
        hit_rate@1 as a float, or None if data not available
    """
    # Check if hit_rate data exists in the results
    hits = 0
    total = 0

    for item in results:
        # Check various possible locations for hit_rate data
        if 'hit_rate' in item:
            return item.get('hit_rate')  # If pre-computed
        if 'memory_hit' in item:
            hits += 1 if item['memory_hit'] else 0
            total += 1
        elif 'searches' in item and 'memory_hit' in item.get('searches', {}):
            hits += 1 if item['searches']['memory_hit'] else 0
            total += 1

    if total > 0:
        return hits / total
    return None  # Data not available


def extract_labels(results: list[dict]) -> tuple[list[int], list[int]]:
    """
    Extract ground truth and predicted labels.

    Returns:
        y_true: Ground truth labels (1 for SUPPORTED, 0 for NOT_SUPPORTED)
        y_pred: Predicted labels (1 for True, 0 for False)
    """
    y_true = []
    y_pred = []

    for item in results:
        # Ground truth
        label = item['label']
        if label == 'SUPPORTED':
            y_true.append(1)
        elif label == 'NOT_SUPPORTED':
            y_true.append(0)
        else:
            print(f"Warning: Unknown label '{label}'")
            continue

        # Predicted
        answer = item['result']['answer']
        if answer == 'True':
            y_pred.append(1)
        elif answer == 'False':
            y_pred.append(0)
        else:
            print(f"Warning: Unknown answer '{answer}'")
            y_true.pop()  # Remove the corresponding ground truth
            continue

    return y_true, y_pred


def evaluate(filepath: str) -> dict:
    """Evaluate a single results file."""
    results = load_results(filepath)
    y_true, y_pred = extract_labels(results)

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    # Calculate per-class metrics
    tn, fp, fn, tp = cm.ravel()

    # Calculate hit_rate@1 if available
    hit_rate = calculate_hit_rate(results)

    return {
        'total': len(y_true),
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'confusion_matrix': cm,
        'true_positives': tp,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'supported_recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'not_supported_recall': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'hit_rate_at_1': hit_rate,
    }


def main():
    results_dir = Path('/Users/macbook/Desktop/fire/results')
    ratios = ['0.2', '0.4', '0.6', '0.8']

    print("=" * 90)
    print("Fire Hover Fact-Verification Evaluation Results")
    print("=" * 90)
    print(f"| {'Ratio':<6} | {'Total':<6} | {'Accuracy':<10} | {'Balanced Acc':<12} | {'Hit_Rate@1':<11} | {'SUP Recall':<11} | {'NOT_SUP Recall':<14} |")
    print("|" + "-" * 8 + "|" + "-" * 8 + "|" + "-" * 12 + "|" + "-" * 14 + "|" + "-" * 13 + "|" + "-" * 13 + "|" + "-" * 16 + "|")

    all_results = {}

    for ratio in ratios:
        filepath = results_dir / f'fire_rag_hover_gpt-4o-mini_r{ratio}.jsonl'
        if filepath.exists():
            metrics = evaluate(str(filepath))
            all_results[ratio] = metrics

            hit_rate_str = f"{metrics['hit_rate_at_1']:.4f}" if metrics['hit_rate_at_1'] is not None else "-"
            print(f"| {ratio:<6} | {metrics['total']:<6} | {metrics['accuracy']:<10.4f} | {metrics['balanced_accuracy']:<12.4f} | {hit_rate_str:<11} | {metrics['supported_recall']:<11.4f} | {metrics['not_supported_recall']:<14.4f} |")
        else:
            print(f"| {ratio:<6} | {'N/A':<6} | {'N/A':<10} | {'N/A':<12} | {'-':<11} | {'N/A':<11} | {'N/A':<14} |")

    print("|" + "-" * 8 + "|" + "-" * 8 + "|" + "-" * 12 + "|" + "-" * 14 + "|" + "-" * 13 + "|" + "-" * 13 + "|" + "-" * 16 + "|")

    # Detailed report for each ratio
    print("\n" + "=" * 90)
    print("Detailed Results")
    print("=" * 90)

    for ratio, metrics in all_results.items():
        print(f"\n--- Ratio {ratio} ---")
        print(f"Total samples: {metrics['total']}")
        print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f} ({metrics['balanced_accuracy']*100:.2f}%)")
        hit_rate_str = f"{metrics['hit_rate_at_1']:.4f} ({metrics['hit_rate_at_1']*100:.2f}%)" if metrics['hit_rate_at_1'] is not None else "N/A"
        print(f"Memory Hit_Rate@1: {hit_rate_str}")
        print(f"\nConfusion Matrix:")
        print(f"                  Predicted")
        print(f"                  False    True")
        print(f"Actual NOT_SUP    {metrics['true_negatives']:<8} {metrics['false_positives']}")
        print(f"       SUPPORTED  {metrics['false_negatives']:<8} {metrics['true_positives']}")
        print(f"\nPer-class recall:")
        print(f"  SUPPORTED recall (sensitivity):     {metrics['supported_recall']:.4f}")
        print(f"  NOT_SUPPORTED recall (specificity): {metrics['not_supported_recall']:.4f}")


if __name__ == '__main__':
    main()
