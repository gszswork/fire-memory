"""
RAG-augmented FIRE fact-checking.

For each dataset:
1. Split into train:test based on training_ratio
2. Train set: collect evidence, fetch full texts, build vector RAG database
3. Test set: first query RAG database, if insufficient then retrieve fresh web
"""

import os
import json
import dataclasses
import argparse
import random
import numpy as np
import tqdm
import requests
from bs4 import BeautifulSoup
from typing import Optional
from sentence_transformers import SentenceTransformer, util
import torch

from common.modeling import Model
from common.shared_config import openai_api_key, serper_api_key, anthropic_api_key, random_seed
from eval.fire.verify_atomic_claim import (
    FinalAnswer,
    GoogleSearchResult,
    call_search,
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
        self.documents: list[dict] = []  # {text, claim, query, snippet, data_idx}
        self.embeddings: Optional[torch.Tensor] = None
        self.total_samples: int = 0  # Total samples in dataset

    def add_documents(self, docs: list[dict]):
        """Add documents to the index. Each doc should have 'data_idx' field."""
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

    def save(self, path: str):
        """Save the RAG database."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            'documents': self.documents,
            'total_samples': self.total_samples,
        }
        with open(path, 'w') as f:
            json.dump(data, f)

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


def fetch_full_content(url: str, max_chars: int = 5000) -> str:
    """Fetch full text content from a URL."""
    try:
        response = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove non-content elements
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            tag.decompose()

        text = soup.get_text(separator=' ', strip=True)
        return text[:max_chars]
    except Exception:
        return ""


def generate_search_queries(claim: str, rater: Model, num_queries: int = 3) -> list[str]:
    """Generate search queries for a claim without running full verification."""
    prompt = f"""Generate {num_queries} diverse Google search queries to find evidence about this claim.
Return as a JSON list of strings.

Claim: {claim}

Output format: ["query1", "query2", "query3"]
"""
    response, _ = rater.generate(prompt)

    try:
        # Extract JSON list from response
        import re
        match = re.search(r'\[.*?\]', response, re.DOTALL)
        if match:
            queries = json.loads(match.group())
            return queries[:num_queries]
    except:
        pass

    # Fallback: use claim itself as query
    return [claim]


def collect_evidence_for_rag(claim: str, rater: Model, data_idx: int, num_queries: int = 3) -> list[dict]:
    """
    Collect evidence for RAG database (training phase only).
    No verification - just generate queries and retrieve evidence.

    Args:
        claim: The claim to collect evidence for
        rater: The LLM model
        data_idx: Index of this sample in the shuffled dataset (for visibility control)
        num_queries: Number of search queries to generate
    """
    # Step 1: Generate search queries
    queries = generate_search_queries(claim, rater, num_queries)

    evidence_docs = []
    for query in queries:
        # Step 2: Execute search
        try:
            snippet = call_search(query)
        except Exception:
            continue

        if snippet and snippet != "No good Google Search result was found":
            evidence_docs.append({
                'text': f"Claim: {claim}\nQuery: {query}\nEvidence: {snippet}",
                'claim': claim,
                'query': query,
                'snippet': snippet,
                'data_idx': data_idx,  # Tag with position for visibility control
            })

    return evidence_docs


def verify_claim_with_rag(
        claim: str,
        rater: Model,
        rag: VectorRAG,
        rag_threshold: float = 0.6,
        rag_top_k: int = 5,
        max_data_idx: Optional[int] = None,
        max_steps: int = fire_config.max_steps,
        max_retries: int = fire_config.max_retries,
) -> tuple[FinalAnswer | None, dict, dict | None]:
    """
    Verify a claim using RAG first, then fall back to web search if needed.

    Args:
        claim: The claim to verify
        rater: The LLM model
        rag: The RAG database
        rag_threshold: Minimum similarity for RAG results
        rag_top_k: Number of RAG results to retrieve
        max_data_idx: Only use RAG docs with data_idx < max_data_idx (visibility control)
        max_steps: Maximum web search steps
        max_retries: Maximum retries per step
    """
    search_results = []
    total_usage = {'input_tokens': 0, 'output_tokens': 0}

    # Step 1: Query RAG database first (with visibility control)
    rag_results = rag.search(claim, top_k=rag_top_k, threshold=rag_threshold, max_data_idx=max_data_idx)

    if rag_results:
        # Use RAG evidence as initial knowledge
        rag_evidence = "\n".join([r['snippet'] for r in rag_results if 'snippet' in r])
        search_results.append(GoogleSearchResult(
            query="[RAG Database]",
            result=rag_evidence
        ))

    # Step 2: Try to get answer with RAG evidence
    if search_results:
        answer_or_next_search, usage = final_answer_or_next_search(
            claim, search_results, rater, diverse_prompt=False, tolerance=fire_config.max_tolerance
        )
        if usage:
            total_usage['input_tokens'] += usage['input_tokens']
            total_usage['output_tokens'] += usage['output_tokens']

        if isinstance(answer_or_next_search, FinalAnswer):
            # RAG was sufficient
            search_dicts = {
                'google_searches': [dataclasses.asdict(s) for s in search_results],
                'source': 'rag'
            }
            return answer_or_next_search, search_dicts, total_usage

    # Step 3: Fall back to web search if RAG insufficient
    stop_search = False
    for _ in range(max_steps):
        answer_or_next_search, num_tries = None, 0
        while not answer_or_next_search and num_tries <= max_retries:
            answer_or_next_search, usage = final_answer_or_next_search(
                claim, search_results, rater,
                diverse_prompt=fire_config.diverse_prompt,
                tolerance=fire_config.max_tolerance
            )
            if usage:
                total_usage['input_tokens'] += usage['input_tokens']
                total_usage['output_tokens'] += usage['output_tokens']
            if answer_or_next_search == '_Early_Stop':
                stop_search = True
                break
            num_tries += 1

        if stop_search:
            break
        if answer_or_next_search is None:
            break
        elif isinstance(answer_or_next_search, GoogleSearchResult):
            search_results.append(answer_or_next_search)
        elif isinstance(answer_or_next_search, FinalAnswer):
            search_dicts = {
                'google_searches': [dataclasses.asdict(s) for s in search_results],
                'source': 'web'
            }
            return answer_or_next_search, search_dicts, total_usage

    # Must get final answer
    final_answer, num_tries = None, 0
    while not final_answer and num_tries <= max_retries:
        num_tries += 1
        final_answer, usage = must_get_final_answer(claim, searches=search_results, model=rater)
        if usage:
            total_usage['input_tokens'] += usage['input_tokens']
            total_usage['output_tokens'] += usage['output_tokens']

    search_dicts = {
        'google_searches': [dataclasses.asdict(s) for s in search_results],
        'source': 'mixed'
    }
    return final_answer, search_dicts, total_usage


def main():
    parser = argparse.ArgumentParser(description="Run RAG-augmented fact-checking")
    parser.add_argument('--benchmark', type=str, required=True, help='Benchmark dataset name')
    parser.add_argument('--build_ratio', type=float, default=0.8, help='Max ratio for building RAG (build once)')
    parser.add_argument('--train_ratio', type=float, default=0.3, help='Ratio of visible data for this evaluation')
    parser.add_argument('--rag_threshold', type=float, default=0.6, help='Similarity threshold for RAG retrieval')
    parser.add_argument('--rag_top_k', type=int, default=5, help='Top-k results from RAG')
    parser.add_argument('--num_queries', type=int, default=3, help='Number of search queries per claim in training')
    parser.add_argument('--build_only', action='store_true', help='Only build RAG database, skip evaluation')
    parser.add_argument('--eval_only', action='store_true', help='Only evaluate, load existing RAG')
    args = parser.parse_args()

    # Validate: train_ratio should not exceed build_ratio
    if args.train_ratio > args.build_ratio:
        raise ValueError(f"train_ratio ({args.train_ratio}) cannot exceed build_ratio ({args.build_ratio})")

    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)

    models = ['openai:gpt-4o-mini']
    framework = 'fire_rag'

    for model in models:
        print(f'Running model: {model}')
        rater = Model(model)
        model_name = model.split(':')[-1].split('/')[-1]

        # Load dataset
        data_path = f'datasets/{args.benchmark}/data.jsonl'
        with open(data_path, 'r') as f:
            all_data = [json.loads(line) for line in f]

        # Shuffle once with fixed seed
        random.shuffle(all_data)
        total_samples = len(all_data)

        # RAG database path (one per benchmark, built at max ratio)
        rag_path = f'rag_db/{args.benchmark}_{model_name}.json'
        rag = VectorRAG()

        # Phase 1: Build RAG at build_ratio (do this once)
        if not args.eval_only:
            print(f"\n=== Phase 1: Building RAG (ratio={args.build_ratio}) ===")
            build_idx = int(total_samples * args.build_ratio)
            build_data = all_data[:build_idx]

            all_evidence = []
            for idx, item in enumerate(tqdm.tqdm(build_data, desc="Collecting evidence")):
                claim = item['claim']
                # Tag each doc with its data_idx for visibility control
                evidence_docs = collect_evidence_for_rag(claim, rater, data_idx=idx, num_queries=args.num_queries)
                all_evidence.extend(evidence_docs)

            print(f"Collected {len(all_evidence)} evidence documents")
            rag.add_documents(all_evidence)
            rag.total_samples = total_samples
            rag.save(rag_path)
            print(f"RAG database saved to {rag_path}")

            if args.build_only:
                print("Build complete. Exiting.")
                return

        else:
            print(f"\n=== Loading existing RAG from {rag_path} ===")
            rag.load(rag_path)
            print(f"Loaded {len(rag.documents)} documents")

        # Phase 2: Evaluate with visibility limited to train_ratio
        print(f"\n=== Phase 2: Evaluating (visible_ratio={args.train_ratio}) ===")

        # Test set is always fixed: last 20% (i.e., after build_ratio)
        test_start_idx = int(total_samples * args.build_ratio)
        test_data = all_data[test_start_idx:]

        # Visibility boundary: only evidence from 0 to train_ratio is visible
        visible_idx = int(total_samples * args.train_ratio)

        print(f"Visible evidence: indices 0-{visible_idx-1} (ratio={args.train_ratio})")
        print(f"Unused evidence: indices {visible_idx}-{test_start_idx-1}")
        print(f"Test set: {len(test_data)} samples (fixed, indices {test_start_idx}-{total_samples-1})")

        output_path = f'results/{framework}_{args.benchmark}_{model_name}_r{args.train_ratio}.jsonl'
        os.makedirs('results', exist_ok=True)

        # Load already processed claims
        processed_claims = set()
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                for line in f:
                    try:
                        existing = json.loads(line)
                        processed_claims.add(existing['claim'])
                    except (json.JSONDecodeError, KeyError):
                        continue

        failed_cnt = 0
        rag_success_cnt = 0
        total_usage = {'input_tokens': 0, 'output_tokens': 0}

        with open(output_path, 'a') as fout:
            for item in tqdm.tqdm(test_data, desc="Evaluating"):
                claim = item['claim']
                label = item['label']

                if claim in processed_claims:
                    continue

                result, searches, usage = verify_claim_with_rag(
                    claim, rater, rag,
                    rag_threshold=args.rag_threshold,
                    rag_top_k=args.rag_top_k,
                    max_data_idx=visible_idx,  # Only see evidence from visible portion
                )

                if usage:
                    total_usage['input_tokens'] += usage['input_tokens']
                    total_usage['output_tokens'] += usage['output_tokens']

                if result is None:
                    failed_cnt += 1
                    continue

                if searches.get('source') == 'rag':
                    rag_success_cnt += 1

                fout.write(json.dumps({
                    'claim': claim,
                    'label': label,
                    'result': dataclasses.asdict(result),
                    'searches': searches
                }) + '\n')

        print(f"\n=== Results (train_ratio={args.train_ratio}) ===")
        print(f"Results saved to: {output_path}")
        print(f"Failed claims: {failed_cnt}")
        print(f"RAG-only successes: {rag_success_cnt} / {len(test_data) - failed_cnt}")
        print(f"Total usage: {total_usage}")


if __name__ == '__main__':
    main()
