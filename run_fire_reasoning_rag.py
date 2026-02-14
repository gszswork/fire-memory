"""
Reasoning Chain RAG for Fact-Checking.

Instead of storing raw evidence chunks, this approach stores complete reasoning
chains — the full FIRE verification trajectory (searches, intermediate reasoning,
prediction) plus an LLM judgment analyzing why the chain succeeded or failed.

At test time, similar claims' reasoning experiences guide the model's own
verification process.

Phases:
1. Build RAG DB (training set): run FIRE verification, judge each chain, store in vector DB
2. Test-time inference:
   Phase A — RAG-only: predict using only retrieved reasoning experiences
   Phase B — Web fallback: search with reasoning context injected into prompts
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
    call_search,
    get_sentence_similarity,
)
from eval.fire import config as fire_config
from common import utils

# Set API keys
os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
os.environ['OPENAI_API_KEY'] = openai_api_key
os.environ["SERPER_API_KEY"] = serper_api_key

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

_Factual_LABEL = 'True'
_Non_Factual_LABEL = 'False'

################################################################################
#                           PROMPT TEMPLATES
################################################################################

# 1. Judgment prompt (correct prediction)
_JUDGMENT_CORRECT_PROMPT = """\
You are analyzing a fact-checking reasoning chain that produced a CORRECT prediction.

Claim: {claim}
Ground truth label: {label}
Prediction: {prediction}

Reasoning chain:
{chain}

In 2-3 sentences, analyze what evidence and reasoning strategy led to the correct prediction. \
Focus on which search queries were most effective and what key evidence was decisive."""

# 2. Judgment prompt (incorrect prediction)
_JUDGMENT_INCORRECT_PROMPT = """\
You are analyzing a fact-checking reasoning chain that produced an INCORRECT prediction.

Claim: {claim}
Ground truth label: {label}
Prediction: {prediction}

Reasoning chain:
{chain}

In 2-3 sentences, analyze where the reasoning went wrong. \
Focus on what evidence was missing, which search queries were ineffective, or what logical errors occurred."""

# 3. RAG-only prediction (Phase A)
_RAG_ONLY_PREDICTION_PROMPT = """\
Instructions:
1. You are provided with a STATEMENT to fact-check and SIMILAR CASE EXPERIENCES from previously verified claims.
2. Review the similar cases to understand how similar claims were verified, what evidence was found, and what the outcomes were.
3. Based on these experiences, assess whether you can confidently determine the factual accuracy of the STATEMENT.
4. If the similar cases provide sufficient insight to make a confident judgment, output:
   {{
     "final_answer": "{factual}" or "{non_factual}"
   }}
5. If the similar cases are NOT sufficient (the claim requires different evidence or the cases are not similar enough), output:
   {{
     "insufficient": true
   }}

SIMILAR CASE EXPERIENCES:
{similar_cases}

STATEMENT:
{claim}"""

# 4a. Reasoning-augmented search: final_answer_or_next_search variant
_REASONING_RAG_SEARCH_PROMPT = """\
Instructions:
1. You are provided with a STATEMENT, relevant KNOWLEDGE points, and SIMILAR CASE EXPERIENCES from previously verified claims.
2. Review the SIMILAR CASE EXPERIENCES to understand how similar claims were verified — what search strategies worked, what evidence was decisive, and what pitfalls to avoid.
3. Based on the KNOWLEDGE and guided by the similar case experiences, assess the factual accuracy of the STATEMENT.
4. Before presenting your conclusion, think through the process step-by-step.
   Include a summary of the key points from the KNOWLEDGE as part of your reasoning.
5. If the KNOWLEDGE allows you to confidently make a decision, output the final
   answer as a JSON object in the following format:
   {{
     "final_answer": "{factual}" or "{non_factual}"
   }}
6. If the KNOWLEDGE is insufficient to make a judgment, issue ONE Google Search
   query that could provide additional evidence. Output the search query in JSON
   format, as follows:
   {{
     "search_query": "Your Google search query here"
   }}
7. The query should aim to obtain new information not already present in the
   KNOWLEDGE, specifically helpful for verifying the STATEMENT's accuracy.
   Use the similar case experiences to guide your search strategy.

SIMILAR CASE EXPERIENCES:
{similar_cases}

KNOWLEDGE:
{knowledge}

STATEMENT:
{claim}"""

# 4b. Reasoning-augmented search: must_get_final_answer variant
_REASONING_RAG_MUST_ANSWER_PROMPT = """\
Instructions:
1. You are provided with a STATEMENT, relevant KNOWLEDGE points, and SIMILAR CASE EXPERIENCES from previously verified claims.
2. Review the SIMILAR CASE EXPERIENCES to understand how similar claims were verified.
3. Based on the KNOWLEDGE and guided by the similar case experiences, assess the factual accuracy of the STATEMENT.
4. Before presenting your final answer, think step-by-step and show your reasoning.
5. Your final answer should be either "{factual}" or "{non_factual}".
6. Format your final answer as a JSON object in the following structure:
   {{
     "final_answer": "{factual}" or "{non_factual}"
   }}

SIMILAR CASE EXPERIENCES:
{similar_cases}

KNOWLEDGE:
{knowledge}

STATEMENT:
{claim}"""


################################################################################
#                              VECTOR RAG
################################################################################

class VectorRAG:
    """Simple vector-based RAG using sentence-transformers and cosine similarity."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(model_name).to(device)
        self.documents: list[dict] = []
        self.embeddings: Optional[torch.Tensor] = None
        self.total_samples: int = 0

    def add_documents(self, docs: list[dict]):
        """Add documents to the index. Each doc should have 'text' and 'data_idx' fields."""
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
        """Search for relevant documents with optional visibility control."""
        if not self.documents or self.embeddings is None:
            return []

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
            if isinstance(data, list):
                self.documents = data
                self.total_samples = 0
            else:
                self.documents = data.get('documents', [])
                self.total_samples = data.get('total_samples', 0)
            if self.documents:
                texts = [d['text'] for d in self.documents]
                self.embeddings = self.encoder.encode(texts, convert_to_tensor=True).to(device)


################################################################################
#                        BUILD PHASE FUNCTIONS
################################################################################

def _truncate_words(text: str, max_words: int) -> str:
    """Truncate text to max_words."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return ' '.join(words[:max_words]) + '...'


def verify_atomic_claim_with_chain(
    atomic_claim: str,
    rater: Model,
    max_steps: int = fire_config.max_steps,
    max_retries: int = fire_config.max_retries,
    diverse_prompt: bool = fire_config.diverse_prompt,
    tolerance: int = fire_config.max_tolerance,
) -> tuple[FinalAnswer | None, dict, dict | None, list[GoogleSearchResult], list[str]]:
    """
    Run FIRE verification loop and capture the full reasoning chain.

    Returns:
        (final_answer, search_dicts, total_usage, search_results_list, prior_reasonings)
    """
    search_results = []
    prior_reasonings = []
    total_usage = {'input_tokens': 0, 'output_tokens': 0}

    stop_search = False
    for _ in range(max_steps):
        answer_or_next_search, num_tries = None, 0
        while not answer_or_next_search and num_tries <= max_retries:
            answer_or_next_search, usage, model_response = final_answer_or_next_search_standard(
                atomic_claim, search_results, rater,
                diverse_prompt=diverse_prompt, tolerance=tolerance,
                prior_reasonings=prior_reasonings,
            )
            if usage is not None:
                total_usage['input_tokens'] += usage['input_tokens']
                total_usage['output_tokens'] += usage['output_tokens']
            if answer_or_next_search == '_Early_Stop':
                if model_response:
                    prior_reasonings.append(model_response)
                stop_search = True
                break
            num_tries += 1
        if stop_search:
            break
        if answer_or_next_search is None:
            break
        elif isinstance(answer_or_next_search, GoogleSearchResult):
            if model_response:
                prior_reasonings.append(model_response)
            search_results.append(answer_or_next_search)
        elif isinstance(answer_or_next_search, FinalAnswer):
            search_dicts = {
                'google_searches': [dataclasses.asdict(s) for s in search_results]
            }
            return answer_or_next_search, search_dicts, total_usage, search_results, prior_reasonings

    # Must get final answer
    final_answer, num_tries = None, 0
    while not final_answer and num_tries <= max_retries:
        num_tries += 1
        final_answer, usage = must_get_final_answer_standard(
            atomic_claim, searches=search_results, model=rater,
            prior_reasonings=prior_reasonings,
        )
        if usage is not None:
            total_usage['input_tokens'] += usage['input_tokens']
            total_usage['output_tokens'] += usage['output_tokens']

    search_dicts = {
        'google_searches': [dataclasses.asdict(s) for s in search_results]
    }
    return final_answer, search_dicts, total_usage, search_results, prior_reasonings


def final_answer_or_next_search_standard(
    atomic_claim: str,
    past_searches: list[GoogleSearchResult],
    model: Model,
    diverse_prompt: bool = False,
    tolerance: int = 4,
    prior_reasonings: list[str] | None = None,
) -> tuple[FinalAnswer | GoogleSearchResult | None | str, dict | None, str | None]:
    """Standard FIRE prompt logic with memory (duplicated to avoid modifying shared code)."""
    from eval.fire.verify_atomic_claim import (
        _FINAL_ANSWER_OR_NEXT_SEARCH_WITH_MEMORY_FORMAT,
        _FINAL_ANSWER_OR_NEXT_SEARCH_FORMAT,
        _STATEMENT_PLACEHOLDER,
        _KNOWLEDGE_PLACEHOLDER,
        _MEMORY_PLACEHOLDER,
        _format_memory,
    )

    knowledge = '\n'.join([s.result for s in past_searches])
    knowledge = 'N/A' if not knowledge else knowledge

    if prior_reasonings:
        template = _FINAL_ANSWER_OR_NEXT_SEARCH_WITH_MEMORY_FORMAT
        memory = _format_memory(prior_reasonings)
    else:
        template = _FINAL_ANSWER_OR_NEXT_SEARCH_FORMAT
        memory = None

    full_prompt = template.replace(_STATEMENT_PLACEHOLDER, atomic_claim)
    full_prompt = full_prompt.replace(_KNOWLEDGE_PLACEHOLDER, knowledge)
    if memory:
        full_prompt = full_prompt.replace(_MEMORY_PLACEHOLDER, memory)
    full_prompt = utils.strip_string(full_prompt)

    query_history = [item.query for item in past_searches]
    search_history = [item.result for item in past_searches]

    if diverse_prompt:
        if len(query_history) >= 2:
            full_prompt += "Please pay attention to optimizing the query to make it more diverse and the retrieved knowledge is as different as possible."
        if len(search_history) >= tolerance - 1 and get_sentence_similarity(
            search_history[-1], search_history[-(tolerance - 1):-1], threshold=0.9
        ) >= tolerance - 2:
            full_prompt += "\n\nPlease note! We have detected multiple very similar contents in the Knowledge section. Please optimize your query so that the retrieved knowledge is as different as possible."
        if len(query_history) >= tolerance - 1 and get_sentence_similarity(
            query_history[-1], query_history[-(tolerance - 1):-1], threshold=0.9
        ) >= tolerance - 2:
            full_prompt += "\nPlease note that we have detected very similar content many times in the past query history. Please pay attention to optimizing the query to make it more diverse."

    model_response, usage = model.generate(full_prompt)
    answer_or_next_query = utils.extract_json_from_output(model_response)
    if answer_or_next_query is None:
        return None, None, model_response

    if 'final_answer' in answer_or_next_query:
        final = FinalAnswer(response=model_response, answer=answer_or_next_query['final_answer'])
        return final, usage, model_response

    if 'search_query' in answer_or_next_query:
        query = answer_or_next_query['search_query']
        if (len(query_history) >= (tolerance - 1)
                and get_sentence_similarity(query, query_history[-(tolerance - 1):]) >= tolerance - 1):
            return '_Early_Stop', usage, model_response
        if (len(search_history) >= tolerance
                and get_sentence_similarity(search_history[-1], search_history[-tolerance:-1]) >= tolerance - 1):
            return '_Early_Stop', usage, model_response
        search_result = GoogleSearchResult(query=query, result=call_search(query))
        return search_result, usage, model_response

    return None, None, model_response


def must_get_final_answer_standard(
    atomic_fact: str,
    searches: list[GoogleSearchResult],
    model: Model,
    prior_reasonings: list[str] | None = None,
) -> tuple[FinalAnswer | None, dict | None]:
    """Standard must-get-final-answer with memory (duplicated)."""
    from eval.fire.verify_atomic_claim import (
        _MUST_HAVE_FINAL_ANSWER_WITH_MEMORY_FORMAT,
        _MUST_HAVE_FINAL_ANSWER_FORMAT,
        _STATEMENT_PLACEHOLDER,
        _KNOWLEDGE_PLACEHOLDER,
        _MEMORY_PLACEHOLDER,
        _format_memory,
    )

    knowledge = '\n'.join(search.result for search in searches)

    if prior_reasonings:
        template = _MUST_HAVE_FINAL_ANSWER_WITH_MEMORY_FORMAT
        memory = _format_memory(prior_reasonings)
    else:
        template = _MUST_HAVE_FINAL_ANSWER_FORMAT
        memory = None

    full_prompt = template.replace(_STATEMENT_PLACEHOLDER, atomic_fact)
    full_prompt = full_prompt.replace(_KNOWLEDGE_PLACEHOLDER, knowledge)
    if memory:
        full_prompt = full_prompt.replace(_MEMORY_PLACEHOLDER, memory)
    full_prompt = utils.strip_string(full_prompt)

    try:
        model_response, usage = model.generate(full_prompt)
        if not model_response:
            return None, None
        answer = utils.extract_json_from_output(model_response)
        if not answer or 'final_answer' not in answer:
            return None, None
        final_answer = answer['final_answer']
        if final_answer in [_Factual_LABEL, _Non_Factual_LABEL]:
            return FinalAnswer(response=model_response, answer=final_answer), usage
        return None, None
    except Exception as e:
        print(f"Error in must_get_final_answer_standard: {e}")
        return None, None


def format_reasoning_chain(
    claim: str,
    searches: list[GoogleSearchResult],
    prior_reasonings: list[str],
    final_answer: FinalAnswer,
    label: str,
    max_evidence_words: int = 200,
    max_reasoning_words: int = 150,
) -> str:
    """Format the reasoning chain compactly with truncated evidence/reasoning per step."""
    parts = [f"Claim: {claim}"]

    for i, search in enumerate(searches):
        parts.append(f"\n--- Step {i+1} ---")
        parts.append(f"Search query: {search.query}")
        truncated_result = _truncate_words(search.result, max_evidence_words)
        parts.append(f"Evidence: {truncated_result}")

        if i < len(prior_reasonings):
            truncated_reasoning = _truncate_words(prior_reasonings[i], max_reasoning_words)
            parts.append(f"Reasoning: {truncated_reasoning}")

    parts.append(f"\n--- Final Answer ---")
    parts.append(f"Prediction: {final_answer.answer}")
    parts.append(f"Ground truth: {label}")

    return '\n'.join(parts)


def judge_reasoning_chain(
    rater: Model,
    claim: str,
    label: str,
    prediction: str,
    chain: str,
    is_correct: bool,
) -> tuple[str, dict | None]:
    """LLM call to analyze why the reasoning chain succeeded or failed."""
    if is_correct:
        prompt = _JUDGMENT_CORRECT_PROMPT.format(
            claim=claim, label=label, prediction=prediction, chain=chain,
        )
    else:
        prompt = _JUDGMENT_INCORRECT_PROMPT.format(
            claim=claim, label=label, prediction=prediction, chain=chain,
        )

    judgment_text, usage = rater.generate(prompt)
    return judgment_text, usage


def build_reasoning_doc(
    claim: str,
    label: str,
    prediction: str,
    is_correct: bool,
    data_idx: int,
    reasoning_chain: str,
    searches: list[GoogleSearchResult],
    prior_reasonings: list[str],
    final_response: str,
    judgment: str,
    judgment_type: str,
) -> dict:
    """Assemble the document dict for the RAG database."""
    return {
        'text': claim,  # Embedding indexed by claim text
        'claim': claim,
        'label': label,
        'prediction': prediction,
        'is_correct': is_correct,
        'data_idx': data_idx,
        'reasoning_chain': reasoning_chain,
        'searches': [{'query': s.query, 'result': s.result} for s in searches],
        'prior_reasonings': prior_reasonings,
        'final_response': final_response,
        'num_steps': len(searches),
        'judgment': judgment,
        'judgment_type': judgment_type,
    }


################################################################################
#                        TEST PHASE FUNCTIONS
################################################################################

def format_similar_cases(retrieved_docs: list[dict]) -> str:
    """Format K retrieved reasoning chains for prompt injection."""
    if not retrieved_docs:
        return "No similar cases found."

    parts = []
    for i, doc in enumerate(retrieved_docs):
        similarity = doc.get('score', 0)
        judgment_type = doc.get('judgment_type', 'unknown')
        outcome = 'CORRECT' if doc.get('is_correct', False) else 'INCORRECT'

        parts.append(f"=== Case {i+1} (similarity: {similarity:.2f}, outcome: {outcome}) ===")
        parts.append(doc.get('reasoning_chain', ''))
        if doc.get('judgment'):
            parts.append(f"Analysis: {doc['judgment']}")
        parts.append("")

    return '\n'.join(parts)


def attempt_rag_only_prediction(
    claim: str,
    similar_cases_text: str,
    rater: Model,
) -> tuple[FinalAnswer | None, dict | None]:
    """
    Phase A: Try to predict using only retrieved reasoning experiences.

    Returns:
        (FinalAnswer, usage) if confident prediction made
        (None, usage) if model says insufficient
    """
    prompt = _RAG_ONLY_PREDICTION_PROMPT.format(
        factual=_Factual_LABEL,
        non_factual=_Non_Factual_LABEL,
        similar_cases=similar_cases_text,
        claim=claim,
    )
    prompt = utils.strip_string(prompt)

    model_response, usage = rater.generate(prompt)
    answer = utils.extract_json_from_output(model_response)

    if answer is None:
        return None, usage

    if 'final_answer' in answer:
        final_answer_val = answer['final_answer']
        if final_answer_val in [_Factual_LABEL, _Non_Factual_LABEL]:
            return FinalAnswer(response=model_response, answer=final_answer_val), usage

    # Model returned insufficient or invalid response
    return None, usage


def _final_answer_or_next_search_with_rag_context(
    atomic_claim: str,
    past_searches: list[GoogleSearchResult],
    model: Model,
    similar_cases_text: str,
    diverse_prompt: bool = False,
    tolerance: int = 4,
) -> tuple[FinalAnswer | GoogleSearchResult | None | str, dict | None, str | None]:
    """FIRE search prompt with reasoning RAG context injected."""
    knowledge = '\n'.join([s.result for s in past_searches])
    knowledge = 'N/A' if not knowledge else knowledge

    full_prompt = _REASONING_RAG_SEARCH_PROMPT.format(
        factual=_Factual_LABEL,
        non_factual=_Non_Factual_LABEL,
        similar_cases=similar_cases_text,
        knowledge=knowledge,
        claim=atomic_claim,
    )
    full_prompt = utils.strip_string(full_prompt)

    query_history = [item.query for item in past_searches]
    search_history = [item.result for item in past_searches]

    if diverse_prompt:
        if len(query_history) >= 2:
            full_prompt += "Please pay attention to optimizing the query to make it more diverse and the retrieved knowledge is as different as possible."
        if len(search_history) >= tolerance - 1 and get_sentence_similarity(
            search_history[-1], search_history[-(tolerance - 1):-1], threshold=0.9
        ) >= tolerance - 2:
            full_prompt += "\n\nPlease note! We have detected multiple very similar contents in the Knowledge section. Please optimize your query so that the retrieved knowledge is as different as possible."
        if len(query_history) >= tolerance - 1 and get_sentence_similarity(
            query_history[-1], query_history[-(tolerance - 1):-1], threshold=0.9
        ) >= tolerance - 2:
            full_prompt += "\nPlease note that we have detected very similar content many times in the past query history. Please pay attention to optimizing the query to make it more diverse."

    model_response, usage = model.generate(full_prompt)
    answer_or_next_query = utils.extract_json_from_output(model_response)
    if answer_or_next_query is None:
        return None, None, model_response

    if 'final_answer' in answer_or_next_query:
        final = FinalAnswer(response=model_response, answer=answer_or_next_query['final_answer'])
        return final, usage, model_response

    if 'search_query' in answer_or_next_query:
        query = answer_or_next_query['search_query']
        if (len(query_history) >= (tolerance - 1)
                and get_sentence_similarity(query, query_history[-(tolerance - 1):]) >= tolerance - 1):
            return '_Early_Stop', usage, model_response
        if (len(search_history) >= tolerance
                and get_sentence_similarity(search_history[-1], search_history[-tolerance:-1]) >= tolerance - 1):
            return '_Early_Stop', usage, model_response
        search_result = GoogleSearchResult(query=query, result=call_search(query))
        return search_result, usage, model_response

    return None, None, model_response


def _must_get_final_answer_with_rag_context(
    atomic_fact: str,
    searches: list[GoogleSearchResult],
    model: Model,
    similar_cases_text: str,
) -> tuple[FinalAnswer | None, dict | None]:
    """Force final answer with reasoning RAG context."""
    knowledge = '\n'.join(search.result for search in searches)

    full_prompt = _REASONING_RAG_MUST_ANSWER_PROMPT.format(
        factual=_Factual_LABEL,
        non_factual=_Non_Factual_LABEL,
        similar_cases=similar_cases_text,
        knowledge=knowledge,
        claim=atomic_fact,
    )
    full_prompt = utils.strip_string(full_prompt)

    try:
        model_response, usage = model.generate(full_prompt)
        if not model_response:
            return None, None
        answer = utils.extract_json_from_output(model_response)
        if not answer or 'final_answer' not in answer:
            return None, None
        final_answer = answer['final_answer']
        if final_answer in [_Factual_LABEL, _Non_Factual_LABEL]:
            return FinalAnswer(response=model_response, answer=final_answer), usage
        return None, None
    except Exception as e:
        print(f"Error in _must_get_final_answer_with_rag_context: {e}")
        return None, None


def verify_claim_with_reasoning_rag(
    claim: str,
    rater: Model,
    similar_cases_text: str,
    max_steps: int = fire_config.max_steps,
    max_retries: int = fire_config.max_retries,
) -> tuple[FinalAnswer | None, dict, dict | None]:
    """
    Phase B: Web search with reasoning context injected into every prompt.

    Returns:
        (final_answer, search_dicts, total_usage)
    """
    search_results = []
    total_usage = {'input_tokens': 0, 'output_tokens': 0}

    stop_search = False
    for _ in range(max_steps):
        answer_or_next_search, num_tries = None, 0
        while not answer_or_next_search and num_tries <= max_retries:
            answer_or_next_search, usage, _ = _final_answer_or_next_search_with_rag_context(
                claim, search_results, rater, similar_cases_text,
                diverse_prompt=fire_config.diverse_prompt,
                tolerance=fire_config.max_tolerance,
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
                'source': 'reasoning_rag+web',
            }
            return answer_or_next_search, search_dicts, total_usage

    # Must get final answer
    final_answer, num_tries = None, 0
    while not final_answer and num_tries <= max_retries:
        num_tries += 1
        final_answer, usage = _must_get_final_answer_with_rag_context(
            claim, search_results, rater, similar_cases_text,
        )
        if usage:
            total_usage['input_tokens'] += usage['input_tokens']
            total_usage['output_tokens'] += usage['output_tokens']

    search_dicts = {
        'google_searches': [dataclasses.asdict(s) for s in search_results],
        'source': 'reasoning_rag+web',
    }
    return final_answer, search_dicts, total_usage


################################################################################
#                                MAIN
################################################################################

def main():
    parser = argparse.ArgumentParser(description="Run reasoning chain RAG fact-checking")
    parser.add_argument('--benchmark', type=str, required=True, help='Benchmark dataset name')
    parser.add_argument('--model', type=str, default='openai:gpt-4o-mini', help='Model name (org:model_id)')
    parser.add_argument('--build_ratio', type=float, default=0.8, help='Max ratio for building RAG (build once)')
    parser.add_argument('--train_ratio', type=float, default=0.3, help='Ratio of visible data for this evaluation')
    parser.add_argument('--rag_threshold', type=float, default=0.4, help='Similarity threshold for RAG retrieval')
    parser.add_argument('--rag_top_k', type=int, default=5, help='Top-k results from RAG')
    parser.add_argument('--build_only', action='store_true', help='Only build RAG database, skip evaluation')
    parser.add_argument('--eval_only', action='store_true', help='Only evaluate, load existing RAG')
    parser.add_argument('--skip_judgment', action='store_true', help='Skip LLM judgment during build (saves API calls)')
    parser.add_argument('--rag_only', action='store_true', help='At test time, no web fallback (Phase A only + forced answer)')
    parser.add_argument('--max_evidence_words', type=int, default=200, help='Truncation per evidence step in stored chain')
    parser.add_argument('--max_reasoning_words', type=int, default=150, help='Truncation per reasoning step in stored chain')
    args = parser.parse_args()

    if args.train_ratio > args.build_ratio:
        raise ValueError(f"train_ratio ({args.train_ratio}) cannot exceed build_ratio ({args.build_ratio})")

    # Set random seed
    random.seed(random_seed)
    np.random.seed(random_seed)

    model_name_full = args.model
    model_name = model_name_full.split(':')[-1].split('/')[-1]
    framework = 'fire_reasoning_rag'

    print(f'Running model: {model_name_full}')
    rater = Model(model_name_full, temperature=0)

    # Load dataset
    data_path = f'datasets/{args.benchmark}/data.jsonl'
    with open(data_path, 'r') as f:
        all_data = [json.loads(line) for line in f]

    # Shuffle once with fixed seed
    random.shuffle(all_data)
    total_samples = len(all_data)

    # RAG database path
    rag_path = f'rag_db/{args.benchmark}_{model_name}_reasoning.json'
    rag = VectorRAG()

    ########################################################################
    # Phase 1: Build RAG DB
    ########################################################################
    if not args.eval_only:
        print(f"\n=== Phase 1: Building Reasoning RAG (ratio={args.build_ratio}) ===")
        build_idx = int(total_samples * args.build_ratio)
        build_data = all_data[:build_idx]

        # Resume support: load existing documents and track processed claims
        existing_claims = set()
        existing_docs = []
        if os.path.exists(rag_path):
            try:
                with open(rag_path, 'r') as f:
                    saved_data = json.load(f)
                if isinstance(saved_data, dict):
                    existing_docs = saved_data.get('documents', [])
                else:
                    existing_docs = saved_data
                existing_claims = {d['claim'] for d in existing_docs}
                print(f"Resuming: {len(existing_claims)} claims already in RAG DB")
            except (json.JSONDecodeError, KeyError):
                pass

        all_docs = list(existing_docs)
        build_usage = {'input_tokens': 0, 'output_tokens': 0}
        correct_cnt, incorrect_cnt, failed_cnt = 0, 0, 0

        for idx, item in enumerate(tqdm.tqdm(build_data, desc="Building reasoning RAG")):
            claim = item['claim']
            label = item['label']

            if claim in existing_claims:
                continue

            # Step 1: Run FIRE verification with chain capture
            final_answer, search_dicts, usage, search_results, prior_reasonings = \
                verify_atomic_claim_with_chain(claim, rater)

            if usage:
                build_usage['input_tokens'] += usage['input_tokens']
                build_usage['output_tokens'] += usage['output_tokens']

            if final_answer is None:
                failed_cnt += 1
                continue

            # Step 2: Compare prediction against ground truth
            prediction = final_answer.answer
            is_correct = prediction.lower() == label.lower()
            if is_correct:
                correct_cnt += 1
            else:
                incorrect_cnt += 1

            # Step 3: Format reasoning chain
            chain = format_reasoning_chain(
                claim, search_results, prior_reasonings, final_answer, label,
                max_evidence_words=args.max_evidence_words,
                max_reasoning_words=args.max_reasoning_words,
            )

            # Step 4: LLM judgment
            judgment = ''
            judgment_type = 'success_analysis' if is_correct else 'failure_analysis'
            if not args.skip_judgment:
                judgment, j_usage = judge_reasoning_chain(
                    rater, claim, label, prediction, chain, is_correct,
                )
                if j_usage:
                    build_usage['input_tokens'] += j_usage['input_tokens']
                    build_usage['output_tokens'] += j_usage['output_tokens']

            # Step 5: Build document
            doc = build_reasoning_doc(
                claim=claim,
                label=label,
                prediction=prediction,
                is_correct=is_correct,
                data_idx=idx,
                reasoning_chain=chain,
                searches=search_results,
                prior_reasonings=prior_reasonings,
                final_response=final_answer.response,
                judgment=judgment,
                judgment_type=judgment_type,
            )
            all_docs.append(doc)
            existing_claims.add(claim)

            # Periodic save every 50 claims
            if len(all_docs) % 50 == 0:
                _save_rag_db(rag_path, all_docs, total_samples)
                print(f"  [checkpoint] Saved {len(all_docs)} docs to {rag_path}")

        # Final save
        _save_rag_db(rag_path, all_docs, total_samples)

        print(f"\n=== Build Summary ===")
        print(f"Total docs: {len(all_docs)}")
        print(f"Correct predictions: {correct_cnt}, Incorrect: {incorrect_cnt}, Failed: {failed_cnt}")
        print(f"Build usage: {build_usage}")
        print(f"RAG DB saved to: {rag_path}")

        # Load into VectorRAG for eval
        rag.documents = all_docs
        rag.total_samples = total_samples
        if all_docs:
            texts = [d['text'] for d in all_docs]
            rag.embeddings = rag.encoder.encode(texts, convert_to_tensor=True).to(device)

        if args.build_only:
            print("Build complete. Exiting.")
            return

    else:
        print(f"\n=== Loading existing RAG from {rag_path} ===")
        rag.load(rag_path)
        print(f"Loaded {len(rag.documents)} documents")

    ########################################################################
    # Phase 2: Evaluate
    ########################################################################
    print(f"\n=== Phase 2: Evaluating (visible_ratio={args.train_ratio}) ===")

    test_start_idx = int(total_samples * args.build_ratio)
    test_data = all_data[test_start_idx:]
    visible_idx = int(total_samples * args.train_ratio)

    print(f"Visible reasoning chains: indices 0-{visible_idx-1} (ratio={args.train_ratio})")
    print(f"Test set: {len(test_data)} samples (indices {test_start_idx}-{total_samples-1})")
    print(f"Mode: {'RAG-only (no web fallback)' if args.rag_only else 'RAG + web fallback'}")

    output_dir = f'results/{framework}_{args.benchmark}'
    os.makedirs(output_dir, exist_ok=True)
    output_path = f'{output_dir}/{framework}_{args.benchmark}_{model_name}_r{args.train_ratio}.jsonl'

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
        print(f"Resuming: {len(processed_claims)} claims already evaluated")

    failed_cnt = 0
    rag_only_cnt = 0
    web_fallback_cnt = 0
    total_usage = {'input_tokens': 0, 'output_tokens': 0}

    with open(output_path, 'a') as fout:
        for item in tqdm.tqdm(test_data, desc="Evaluating"):
            claim = item['claim']
            label = item['label']

            if claim in processed_claims:
                continue

            # Retrieve similar reasoning chains
            rag_results = rag.search(
                claim, top_k=args.rag_top_k, threshold=args.rag_threshold,
                max_data_idx=visible_idx,
            )
            similar_cases_text = format_similar_cases(rag_results)

            # Phase A: RAG-only attempt
            result, usage = attempt_rag_only_prediction(claim, similar_cases_text, rater)
            if usage:
                total_usage['input_tokens'] += usage.get('input_tokens', 0)
                total_usage['output_tokens'] += usage.get('output_tokens', 0)

            source = 'reasoning_rag'
            searches = {'google_searches': [], 'source': source}

            if result is not None:
                rag_only_cnt += 1
            elif args.rag_only:
                # Force answer with RAG context only (no web search)
                result, usage = _must_get_final_answer_with_rag_context(
                    claim, [], rater, similar_cases_text,
                )
                if usage:
                    total_usage['input_tokens'] += usage.get('input_tokens', 0)
                    total_usage['output_tokens'] += usage.get('output_tokens', 0)
                source = 'reasoning_rag_forced'
                searches = {'google_searches': [], 'source': source}
                if result is not None:
                    rag_only_cnt += 1
            else:
                # Phase B: Web search with reasoning context
                result, searches, usage = verify_claim_with_reasoning_rag(
                    claim, rater, similar_cases_text,
                )
                if usage:
                    total_usage['input_tokens'] += usage.get('input_tokens', 0)
                    total_usage['output_tokens'] += usage.get('output_tokens', 0)
                source = searches.get('source', 'reasoning_rag+web')
                web_fallback_cnt += 1

            if result is None:
                failed_cnt += 1
                continue

            output_record = {
                'claim': claim,
                'label': label,
                'result': dataclasses.asdict(result),
                'searches': searches,
                'source': source,
                'rag_retrieved': len(rag_results),
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
    print(f"RAG-only predictions: {rag_only_cnt}")
    print(f"Web fallback predictions: {web_fallback_cnt}")
    print(f"Failed: {failed_cnt}")
    print(f"Total tokens: input={total_usage['input_tokens']}, output={total_usage['output_tokens']}")

    # Save summary
    summary_path = f'{output_dir}/{framework}_{args.benchmark}_{model_name}_r{args.train_ratio}_summary.json'
    summary = {
        'benchmark': args.benchmark,
        'model': model_name,
        'train_ratio': args.train_ratio,
        'build_ratio': args.build_ratio,
        'rag_top_k': args.rag_top_k,
        'rag_threshold': args.rag_threshold,
        'rag_only': args.rag_only,
        'total_test_samples': len(test_data),
        'successful': successful,
        'rag_only_predictions': rag_only_cnt,
        'web_fallback_predictions': web_fallback_cnt,
        'failed': failed_cnt,
        'total_usage': total_usage,
        'method': 'reasoning_chain_rag',
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")


def _save_rag_db(path: str, docs: list[dict], total_samples: int):
    """Helper to save RAG DB with periodic checkpoints."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {'documents': docs, 'total_samples': total_samples}
    with open(path, 'w') as f:
        json.dump(data, f)


if __name__ == '__main__':
    main()
