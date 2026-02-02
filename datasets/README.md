# Fact-Checking Dataset

This dataset contains four separate data folders, each representing a different fact-checking dataset. All datasets share the same structure, and the label is a binary indicator of whether the claim is true or false.

## Structure

The dataset consists of the following four directories:

1. **bingcheck**
2. **factcheckbench**
3. **factool_qa**
4. **felm_wk**

Each directory contains a `data.jsonl` file, which holds the data in the following format:

### Format
Each `.jsonl` file (JSON Lines format) contains multiple lines where each line is a JSON object with two keys:
- **claim**: A string representing the claim to be verified.
- **label**: A binary label indicating the veracity of the claim. This can be:
  - `true`: The claim is factual.
  - `false`: The claim is not factual.

### Example Entry

```json
{
    "claim": "The Eiffel Tower is located in Berlin.",
    "label": "false"
}
```

### Usage

To use this dataset, you can load each `.jsonl` file using a JSON library in Python, such as:

```python
import json

# Example: Loading data from bingcheck/data.jsonl
with open('bingcheck/data.jsonl', 'r') as file:
    for line in file:
        data = json.loads(line)
        claim = data['claim']
        label = data['label']
        print(f"Claim: {claim} | Label: {label}")
```
