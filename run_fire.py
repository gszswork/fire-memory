import os
import json
import dataclasses
import tqdm
from langchain_community.callbacks.manager import get_openai_callback
from common.modeling import Model
from common.shared_config import openai_api_key, serper_api_key, anthropic_api_key
from common.utils import calculate_cost_claude

# Set API keys for various services
os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
os.environ['OPENAI_API_KEY'] = openai_api_key
os.environ["SERPER_API_KEY"] = serper_api_key

# List of models to use for the benchmark
models = ['openai:gpt-4o-mini']  # Can include other models like Claude, Llama, Mistral, etc.
# ['gpt-4o-mini', 'gpt-4o', 'o1-preview', 'o1-mini', 'claude-3-haiku-20240307', 'claude-3-opus-20240229', 'claude-3-5-sonnet-20240620', 'Meta-Llama-3.1-8B-Instruct', 'Mistral-7B-Instruct-v0.3',]
# Define benchmark dataset and framework type
# benchmark = 'factcheckbench'  # e.g., 'bingcheck', 'factool_qa', etc.
import argparse

parser = argparse.ArgumentParser(description="Run fact-checking benchmark")
parser.add_argument('--benchmark', type=str, required=True, help='Name of the benchmark dataset to use')
arg = parser.parse_args()

benchmark = arg.benchmark
framework = 'fire'  # Could be 'safe' or 'fire', depending on the task

# Import the appropriate fact-checking function based on the framework
if framework == 'fire':
    from eval.fire.verify_atomic_claim import verify_atomic_claim
# elif framework == 'safe':
#     from eval.safe.rate_atomic_fact import check_atomic_fact

for model in models:
    with get_openai_callback() as cb:
        print(f'Running model: {model}')
        rater = Model(model)
        failed_cnt = 0
        model_name = model.split(':')[-1].split('/')[-1]  # Extract model name for file saving

        # Initialize total token usage tracking
        total_usage = {
            'input_tokens': 0,
            'output_tokens': 0,
        }

        # Load existing claims from output file if it exists
        output_file_path = f'results/{framework}_{benchmark}_{model_name}.jsonl'
        processed_claims = set()
        if os.path.exists(output_file_path):
            with open(output_file_path, 'r') as fin:
                for line in fin:
                    try:
                        existing_data = json.loads(line)
                        processed_claims.add(existing_data['claim'])
                    except (json.JSONDecodeError, KeyError):
                        continue

        # Open output file to append the results
        with open(output_file_path, 'a') as fout:
            # Read and process each line from the dataset
            for line in tqdm.tqdm(open(f'datasets/{benchmark}/data.jsonl', 'r').readlines()):
                data = json.loads(line)  # Load JSON data

                claim = data['claim']    # Extract the claim
                label = data['label']    # Extract the label

                # Skip if claim already processed
                if claim in processed_claims:
                    continue

                # Run the fact-checking function for the claim
                result, searches, usage = verify_atomic_claim(claim, rater)

                # Track token usage if available
                if usage is not None:
                    total_usage['input_tokens'] += usage['input_tokens']
                    total_usage['output_tokens'] += usage['output_tokens']

                # If result is None, count it as a failure and skip further processing
                if result is None:
                    failed_cnt += 1
                    continue

                # Write the result to the output file
                fout.write(json.dumps({
                    'claim': claim,
                    'label': label,
                    'result': dataclasses.asdict(result),
                    'searches': searches
                }) + '\n')
            
        print(f'All fact chekcing results saved to file: results/{framework}_{benchmark}_{model_name}.jsonl')
        # Print the count of failed claims and usage callback
        print(f'Failed claims: {failed_cnt}')
        print(cb)



        