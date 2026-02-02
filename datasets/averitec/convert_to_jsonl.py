#!/usr/bin/env python3

import json

def convert_averitec_to_jsonl():
    # Read the input JSON file
    with open('dev.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Write to JSONL format
    with open('data.jsonl', 'w', encoding='utf-8') as f:
        for item in data:
            simplified_item = {
                'claim': item['claim'],
                'label': item['label']
            }
            f.write(json.dumps(simplified_item) + '\n')

    print(f"Converted {len(data)} items to data.jsonl")

if __name__ == "__main__":
    convert_averitec_to_jsonl()