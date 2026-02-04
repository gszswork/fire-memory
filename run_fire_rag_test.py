"""
RAG-only test evaluation script.

Tests on the fixed test set (last 20% of data) using only RAG database knowledge.
No web retrieval is used - this tests the model's reasoning ability on unseen samples
with varying amounts of RAG knowledge (controlled by train_ratio).

Ratios:
- train_ratio=0: No RAG evidence visible (baseline)
- train_ratio=0.2: Evidence from first 20% of training data visible
- train_ratio=0.4: Evidence from first 40% of training data visible
- train_ratio=0.6: Evidence from first 60% of training data visible
- train_ratio=0.8: Evidence from first 80% of training data visible (max)
"""

import os
import json
import dataclasses
import argparse
import random
import numpy as np
import tqdm
from typing import Optional
from sentence_transformers import SentenceTransformer, util
import torch

from common.modeling import Model
from common.shared_config import openai_api_key, serper_api_key, anthropic_api_key, random_seed
from eval.fire.verify_atomic_claim import (
    FinalAnswer,
    GoogleSearchResult,
    final_answer_or_next_search,
    must_get_final_answer,
)
from eval.fire import config as fire_config

# Set API keys
os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
os.environ['OPENAI_API_KEY'] = openai_api_key
os.environ["SERPER_API_KEY"] = serper_api_key

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"


class VectorRAG:
    """Simple vector-based RAG using sentence-transformers and FAISS-like cosine similarity."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(model_name).to(device)
        self.documents: list[dict] = []
        self.embeddings: Optional[torch.Tensor] = None
        self.total_samples: int = 0

    def add_documents(self, docs: list[dict]):
        """Add documents to the index."""
        for doc in docs:
            self.documents.append(doc)

        if self.documents:
            texts = [d['text'] for d in self.documents]
            self.embeddings = self.encoder.encode(texts, convert_to_tensor=True).to(device)

    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.5,
        max_data_idx: Optional[int] = None,
    ) -> list[dict]:
        """
        Search for relevant documents.

        Args:
            query: Search query
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            max_data_idx: Only return docs with data_idx < max_data_idx (for visibility control)
        """
        if not self.documents or self.embeddings is None:
            return []

        # Filter by data_idx if specified
        if max_data_idx is not None:
            valid_indices = [i for i, d in enumerate(self.documents) if d.get('data_idx', 0) < max_data_idx]
            if not valid_indices:
                return []
            valid_embeddings = self.embeddings[valid_indices]
        else:
            valid_indices = list(range(len(self.documents)))
            valid_embeddings = self.embeddings

        query_embedding = self.encoder.encode(query, convert_to_tensor=True).to(device)
        similarities = util.cos_sim(query_embedding, valid_embeddings)[0]

        # Get top-k results above threshold
        top_results = []
        k = min(top_k, len(valid_indices))
        scores, indices = torch.topk(similarities, k)

        for score, idx in zip(scores, indices):
            if score.item() >= threshold:
                original_idx = valid_indices[idx.item()]
                doc = self.documents[original_idx].copy()
                doc['score'] = score.item()
                top_results.append(doc)

        return top_results

    def load(self, path: str):
        """Load the RAG database."""
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            # Handle both old and new format
            if isinstance(data, list):
                self.documents = data
                self.total_samples = 0
            else:
                self.documents = data.get('documents', [])
                self.total_samples = data.get('total_samples', 0)
            if self.documents:
                texts = [d['text'] for d in self.documents]
                self.embeddings = self.encoder.encode(texts, convert_to_tensor=True).to(device)


def verify_claim_rag_only(
        claim: str,
        rater: Model,
        rag: VectorRAG,
        rag_threshold: float = 0.4,
        rag_top_k: int = 5,
        max_data_idx: Optional[int] = None,
        max_retries: int = fire_config.max_retries,
) -> tuple[FinalAnswer | None, dict, dict | None, dict]:
    """
    Verify a claim using ONLY RAG database - NO web retrieval.

    This tests the model's reasoning ability on unseen samples
    with only the knowledge stored in the RAG database.

    Args:
        claim: The claim to verify
        rater: The LLM model
        rag: The RAG database
        rag_threshold: Minimum similarity for RAG results
        rag_top_k: Number of RAG results to retrieve
        max_data_idx: Only use RAG docs with data_idx < max_data_idx (visibility control)
        max_retries: Maximum retries per step
    """
    search_results = []
    total_usage = {'input_tokens': 0, 'output_tokens': 0}

    # RAG context tracking
    rag_context = {
        'retrieved_count': 0,
        'concluded_by_rag': False,
        'web_retrieval_disabled': True,  # Mark that web retrieval is disabled
    }

    # Query RAG database (with visibility control)
    rag_results = rag.search(claim, top_k=rag_top_k, threshold=rag_threshold, max_data_idx=max_data_idx)
    rag_context['retrieved_count'] = len(rag_results)

    if rag_results:
        # Use RAG evidence as knowledge
        rag_evidence = "\n".join([r['snippet'] for r in rag_results if 'snippet' in r])
        search_results.append(GoogleSearchResult(
            query="[RAG Database]",
            result=rag_evidence
        ))

    # Try to get answer with RAG evidence only (no web fallback)
    if search_results:
        answer_or_next_search, usage = final_answer_or_next_search(
            claim, search_results, rater, diverse_prompt=False, tolerance=fire_config.max_tolerance
        )
        if usage:
            total_usage['input_tokens'] += usage['input_tokens']
            total_usage['output_tokens'] += usage['output_tokens']

        if isinstance(answer_or_next_search, FinalAnswer):
            rag_context['concluded_by_rag'] = True
            search_dicts = {
                'google_searches': [dataclasses.asdict(s) for s in search_results],
                'source': 'rag_only'
            }
            return answer_or_next_search, search_dicts, total_usage, rag_context

    # If RAG insufficient, force a final answer without web search
    # The model must reason with available (or no) evidence
    final_answer, num_tries = None, 0
    while not final_answer and num_tries <= max_retries:
        num_tries += 1
        final_answer, usage = must_get_final_answer(claim, searches=search_results, model=rater)
        if usage:
            total_usage['input_tokens'] += usage['input_tokens']
            total_usage['output_tokens'] += usage['output_tokens']

    search_dicts = {
        'google_searches': [dataclasses.asdict(s) for s in search_results],
        'source': 'rag_only_forced'
    }
    rag_context['concluded_by_rag'] = False
    return final_answer, search_dicts, total_usage, rag_context


def main():
    parser = argparse.ArgumentParser(description="Test RAG-only fact-checking (no web retrieval)")
    parser.add_argument('--benchmark', type=str, required=True, help='Benchmark dataset name')
    parser.add_argument('--build_ratio', type=float, default=0.8, help='Ratio used to build RAG (for test set split)')
    parser.add_argument('--train_ratio', type=float, default=0.0, help='Ratio of visible RAG data for this evaluation')
    parser.add_argument('--rag_threshold', type=float, default=0.4, help='Similarity threshold for RAG retrieval')
    parser.add_argument('--rag_top_k', type=int, default=5, help='Top-k results from RAG')
    args = parser.parse_args()

    # Validate
    if args.train_ratio > args.build_ratio:
        raise ValueError(f"train_ratio ({args.train_ratio}) cannot exceed build_ratio ({args.build_ratio})")

    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)

    models = ['openai:gpt-4o-mini']
    framework = 'fire_rag_test'

    for model in models:
        print(f'Running model: {model}')
        rater = Model(model)
        model_name = model.split(':')[-1].split('/')[-1]

        # Load dataset
        data_path = f'datasets/{args.benchmark}/data.jsonl'
        with open(data_path, 'r') as f:
            all_data = [json.loads(line) for line in f]

        # Shuffle with fixed seed (same as during build)
        random.shuffle(all_data)
        total_samples = len(all_data)

        # Load RAG database
        rag_path = f'rag_db/{args.benchmark}_{model_name}.json'
        print(f"\n=== Loading RAG from {rag_path} ===")
        rag = VectorRAG()

        if not os.path.exists(rag_path):
            raise FileNotFoundError(f"RAG database not found at {rag_path}. Please run build first.")

        rag.load(rag_path)
        print(f"Loaded {len(rag.documents)} documents")

        # Test set: always last 20% (after build_ratio)
        test_start_idx = int(total_samples * args.build_ratio)
        test_data = all_data[test_start_idx:]

        # Visibility boundary: only evidence from 0 to train_ratio is visible
        visible_idx = int(total_samples * args.train_ratio)

        print(f"\n=== Test Configuration ===")
        print(f"Total samples: {total_samples}")
        print(f"Test set: {len(test_data)} samples (indices {test_start_idx}-{total_samples-1})")
        print(f"Visible RAG evidence: indices 0-{visible_idx-1 if visible_idx > 0 else 'NONE'} (train_ratio={args.train_ratio})")
        print(f"Web retrieval: DISABLED")

        # Output path with train_ratio
        output_dir = 'results_test'
        os.makedirs(output_dir, exist_ok=True)
        output_path = f'{output_dir}/{framework}_{args.benchmark}_{model_name}_r{args.train_ratio}.jsonl'

        # Load already processed claims (for resume capability)
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

        # Statistics
        failed_cnt = 0
        rag_concluded_cnt = 0
        total_usage = {'input_tokens': 0, 'output_tokens': 0}
        results_list = []

        with open(output_path, 'a') as fout:
            for item in tqdm.tqdm(test_data, desc=f"Testing (ratio={args.train_ratio})"):
                claim = item['claim']
                label = item['label']

                if claim in processed_claims:
                    continue

                result, searches, usage, rag_context = verify_claim_rag_only(
                    claim, rater, rag,
                    rag_threshold=args.rag_threshold,
                    rag_top_k=args.rag_top_k,
                    max_data_idx=visible_idx if visible_idx > 0 else None,
                )

                if usage:
                    total_usage['input_tokens'] += usage['input_tokens']
                    total_usage['output_tokens'] += usage['output_tokens']

                if result is None:
                    failed_cnt += 1
                    continue

                if rag_context.get('concluded_by_rag'):
                    rag_concluded_cnt += 1

                output_record = {
                    'claim': claim,
                    'label': label,
                    'result': dataclasses.asdict(result),
                    'searches': searches,
                    'rag_context': rag_context,
                    'train_ratio': args.train_ratio,
                }
                fout.write(json.dumps(output_record) + '\n')
                results_list.append(output_record)

        # Summary statistics
        total_evaluated = len(test_data) - len(processed_claims)
        successful = total_evaluated - failed_cnt

        print(f"\n=== Results Summary (train_ratio={args.train_ratio}) ===")
        print(f"Output file: {output_path}")
        print(f"Total test samples: {len(test_data)}")
        print(f"Evaluated this run: {total_evaluated}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed_cnt}")
        print(f"RAG-concluded: {rag_concluded_cnt} / {successful}")
        print(f"Total tokens: input={total_usage['input_tokens']}, output={total_usage['output_tokens']}")

        # Save summary
        summary_path = f'{output_dir}/{framework}_{args.benchmark}_{model_name}_r{args.train_ratio}_summary.json'
        summary = {
            'benchmark': args.benchmark,
            'model': model_name,
            'train_ratio': args.train_ratio,
            'build_ratio': args.build_ratio,
            'total_test_samples': len(test_data),
            'successful': successful,
            'failed': failed_cnt,
            'rag_concluded': rag_concluded_cnt,
            'total_usage': total_usage,
            'web_retrieval': False,
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {summary_path}")


if __name__ == '__main__':
    main()
