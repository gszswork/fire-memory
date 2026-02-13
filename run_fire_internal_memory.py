"""
FIRE internal memory evaluation script.

Tests on the fixed test set (last 20% of data) using FIRE's verify_atomic_claim(),
which performs iterative web search + LLM reasoning with memory (carrying forward
prior chain-of-thought across search iterations).

Test set split is consistent with run_fire_rag.py and run_fire_rag_test.py:
- Data is loaded from datasets/{benchmark}/data.jsonl
- Shuffled with fixed random seed
- build_ratio (default 0.8) determines the split point
- Test set = last (1 - build_ratio) of shuffled data
"""

import os
import json
import dataclasses
import argparse
import random
import numpy as np
import tqdm

from common.modeling import Model
from common.shared_config import openai_api_key, serper_api_key, anthropic_api_key, random_seed
from eval.fire.verify_atomic_claim import verify_atomic_claim, FinalAnswer

# Set API keys
os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
os.environ['OPENAI_API_KEY'] = openai_api_key
os.environ["SERPER_API_KEY"] = serper_api_key


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fact-checking using FIRE's verify_atomic_claim (web search + internal memory)"
    )
    parser.add_argument('--benchmark', type=str, required=True, help='Benchmark dataset name')
    parser.add_argument('--model', type=str, default='openai:gpt-4o-mini', help='Model name (org:model_id)')
    parser.add_argument('--build_ratio', type=float, default=0.8, help='Ratio for test split (consistent with other scripts)')
    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)

    print(f'Running model: {args.model}')
    rater = Model(args.model, temperature=0)
    model_name = args.model.split(':')[-1].split('/')[-1]

    # Load dataset
    data_path = f'datasets/{args.benchmark}/data.jsonl'
    with open(data_path, 'r') as f:
        all_data = [json.loads(line) for line in f]

    # Shuffle with fixed seed (same as run_fire_rag.py and run_fire_rag_test.py)
    random.shuffle(all_data)
    total_samples = len(all_data)

    # Test set: last (1 - build_ratio) of shuffled data
    test_start_idx = int(total_samples * args.build_ratio)
    test_data = all_data[test_start_idx:]

    print(f"\n=== Configuration ===")
    print(f"Benchmark: {args.benchmark}")
    print(f"Model: {args.model}")
    print(f"Total samples: {total_samples}")
    print(f"Test set: {len(test_data)} samples (indices {test_start_idx}-{total_samples-1})")
    print(f"Method: verify_atomic_claim (FIRE with web search + internal memory)")

    # Output
    framework = 'fire_internal_memory'
    output_dir = f'results/{framework}_{args.benchmark}'
    os.makedirs(output_dir, exist_ok=True)
    output_path = f'{output_dir}/{framework}_{args.benchmark}_{model_name}.jsonl'

    # Resume: load already processed claims
    processed_claims = set()
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            for line in f:
                try:
                    existing = json.loads(line)
                    processed_claims.add(existing['claim'])
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"Resuming: {len(processed_claims)} claims already processed")

    # Evaluate
    failed_cnt = 0
    total_usage = {'input_tokens': 0, 'output_tokens': 0}

    with open(output_path, 'a') as fout:
        for item in tqdm.tqdm(test_data, desc=f"Evaluating ({args.benchmark})"):
            claim = item['claim']
            label = item['label']

            if claim in processed_claims:
                continue

            result, searches, usage = verify_atomic_claim(claim, rater)

            if usage:
                total_usage['input_tokens'] += usage['input_tokens']
                total_usage['output_tokens'] += usage['output_tokens']

            if result is None:
                failed_cnt += 1
                continue

            output_record = {
                'claim': claim,
                'label': label,
                'result': dataclasses.asdict(result),
                'searches': searches,
            }
            fout.write(json.dumps(output_record) + '\n')

    # Summary
    total_evaluated = len(test_data) - len(processed_claims)
    successful = total_evaluated - failed_cnt

    print(f"\n=== Results Summary ===")
    print(f"Output file: {output_path}")
    print(f"Total test samples: {len(test_data)}")
    print(f"Evaluated this run: {total_evaluated}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed_cnt}")
    print(f"Total tokens: input={total_usage['input_tokens']}, output={total_usage['output_tokens']}")

    # Save summary
    summary_path = f'{output_dir}/{framework}_{args.benchmark}_{model_name}_summary.json'
    summary = {
        'benchmark': args.benchmark,
        'model': model_name,
        'build_ratio': args.build_ratio,
        'total_test_samples': len(test_data),
        'successful': successful,
        'failed': failed_cnt,
        'total_usage': total_usage,
        'method': 'verify_atomic_claim',
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()
