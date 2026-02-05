"""
Analyze test results from RAG-only evaluation.

Aggregates results across all datasets and ratios, computes accuracy metrics,
and generates summary tables for analysis.
"""

import os
import json
import argparse
from collections import defaultdict
import pandas as pd


def compute_accuracy(results: list[dict]) -> dict:
    """Compute accuracy metrics from results."""
    correct = 0
    total = 0

    label_mapping = {
        'supported': 'supported',
        'refuted': 'refuted',
        'not enough evidence': 'not enough evidence',
        'nei': 'not enough evidence',
        'true': 'supported',
        'false': 'refuted',
        'half-true': 'not enough evidence',
        'mostly-true': 'supported',
        'mostly-false': 'refuted',
        'pants-fire': 'refuted',
        'barely-true': 'refuted',
    }

    for r in results:
        total += 1
        pred_label = r.get('result', {}).get('answer', '').lower()
        true_label = r.get('label', '').lower()

        # Normalize labels
        pred_normalized = label_mapping.get(pred_label, pred_label)
        true_normalized = label_mapping.get(true_label, true_label)

        if pred_normalized == true_normalized:
            correct += 1

    return {
        'correct': correct,
        'total': total,
        'accuracy': correct / total if total > 0 else 0,
    }


def load_results(results_dir: str, benchmark: str, model: str, ratio: float) -> list[dict]:
    """Load results from a specific test run."""
    pattern = f'fire_rag_test_{benchmark}_{model}_r{ratio}.jsonl'
    filepath = os.path.join(results_dir, pattern)

    if not os.path.exists(filepath):
        return []

    results = []
    with open(filepath, 'r') as f:
        for line in f:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze RAG-only test results")
    parser.add_argument('--results_dir', type=str, default='results_test', help='Directory containing results')
    parser.add_argument('--output', type=str, default='results_test/analysis_summary.csv', help='Output CSV path')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', help='Model name')
    args = parser.parse_args()

    # Datasets and ratios
    datasets = ['averitec', 'aggrefact_cnn', 'pubhealth', 'summeval', 'LIAR-New']
    ratios = [0, 0.2, 0.4, 0.6, 0.8]

    # Collect all metrics
    all_metrics = []

    for dataset in datasets:
        for ratio in ratios:
            results = load_results(args.results_dir, dataset, args.model, ratio)

            if not results:
                print(f"Warning: No results found for {dataset} ratio={ratio}")
                continue

            metrics = compute_accuracy(results)

            # Count RAG concluded vs forced
            rag_concluded = sum(1 for r in results if r.get('rag_context', {}).get('concluded_by_rag', False))
            avg_retrieved = sum(r.get('rag_context', {}).get('retrieved_count', 0) for r in results) / len(results) if results else 0

            all_metrics.append({
                'dataset': dataset,
                'train_ratio': ratio,
                'total': metrics['total'],
                'correct': metrics['correct'],
                'accuracy': metrics['accuracy'],
                'rag_concluded': rag_concluded,
                'rag_concluded_pct': rag_concluded / metrics['total'] if metrics['total'] > 0 else 0,
                'avg_retrieved': avg_retrieved,
            })

    if not all_metrics:
        print("No results found. Please run tests first.")
        return

    # Create DataFrame
    df = pd.DataFrame(all_metrics)

    # Save detailed results
    df.to_csv(args.output, index=False)
    print(f"Detailed results saved to: {args.output}")

    # Print summary tables
    print("\n" + "="*80)
    print("ACCURACY BY DATASET AND RATIO")
    print("="*80)

    # Pivot table for accuracy
    pivot = df.pivot(index='dataset', columns='train_ratio', values='accuracy')
    pivot = pivot.round(4)
    print(pivot.to_string())

    # Save pivot table
    pivot_path = args.output.replace('.csv', '_pivot.csv')
    pivot.to_csv(pivot_path)
    print(f"\nPivot table saved to: {pivot_path}")

    # Compute average across datasets
    print("\n" + "="*80)
    print("AVERAGE ACCURACY ACROSS ALL DATASETS")
    print("="*80)
    avg_by_ratio = df.groupby('train_ratio')['accuracy'].mean()
    print(avg_by_ratio.to_string())

    # RAG utilization stats
    print("\n" + "="*80)
    print("RAG CONCLUDED PERCENTAGE BY RATIO")
    print("="*80)
    rag_pivot = df.pivot(index='dataset', columns='train_ratio', values='rag_concluded_pct')
    rag_pivot = rag_pivot.round(4)
    print(rag_pivot.to_string())

    # Average retrieved documents
    print("\n" + "="*80)
    print("AVG RETRIEVED DOCUMENTS BY RATIO")
    print("="*80)
    ret_pivot = df.pivot(index='dataset', columns='train_ratio', values='avg_retrieved')
    ret_pivot = ret_pivot.round(2)
    print(ret_pivot.to_string())

    # Save full analysis
    full_analysis = {
        'accuracy_pivot': pivot.to_dict(),
        'avg_accuracy_by_ratio': avg_by_ratio.to_dict(),
        'rag_concluded_pivot': rag_pivot.to_dict(),
        'avg_retrieved_pivot': ret_pivot.to_dict(),
        'raw_metrics': all_metrics,
    }

    analysis_json_path = args.output.replace('.csv', '.json')
    with open(analysis_json_path, 'w') as f:
        json.dump(full_analysis, f, indent=2)
    print(f"\nFull analysis saved to: {analysis_json_path}")


if __name__ == '__main__':
    main()
