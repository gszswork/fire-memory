"""
Rates a single atomic claim for accuracy.
For each atomic claim, the process would be to prompt the model think of the search term to obtain relevant information,
and then let the model decide if the information is enough to make a judgement or the model needs to continue searching.
"""

import dataclasses
import torch
from typing import Any
from common import modeling, shared_config, utils
from eval.fire import config as fire_config
from eval.fire import query_serper
from sentence_transformers import SentenceTransformer, util

device = "cuda" if torch.cuda.is_available() else "cpu"
sbert_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)
_Factual_LABEL = 'True'
_Non_Factual_LABEL = 'False'
_STATEMENT_PLACEHOLDER = '[STATEMENT]'
_KNOWLEDGE_PLACEHOLDER = '[KNOWLEDGE]'
_MEMORY_PLACEHOLDER = '[MEMORY]'


_FINAL_ANSWER_OR_NEXT_SEARCH_FORMAT = f"""\
Instructions:
1. You are provided with a STATEMENT and relevant KNOWLEDGE points.
2. Based on the KNOWLEDGE, assess the factual accuracy of the STATEMENT.
3. Before presenting your conclusion, think through the process step-by-step. 
   Include a summary of the key points from the KNOWLEDGE as part of your reasoning.
4. If the KNOWLEDGE allows you to confidently make a decision, output the final 
   answer as a JSON object in the following format:
   {{
     "final_answer": "{_Factual_LABEL}" or "{_Non_Factual_LABEL}"
   }}
5. If the KNOWLEDGE is insufficient to make a judgment, issue ONE Google Search 
   query that could provide additional evidence. Output the search query in JSON 
   format, as follows:
   {{
     "search_query": "Your Google search query here"
   }}
6. The query should aim to obtain new information not already present in the 
   KNOWLEDGE, specifically helpful for verifying the STATEMENT's accuracy.

KNOWLEDGE:
{_KNOWLEDGE_PLACEHOLDER}

STATEMENT:
{_STATEMENT_PLACEHOLDER}
"""


_FINAL_ANSWER_OR_NEXT_SEARCH_WITH_MEMORY_FORMAT = f"""\
Instructions:
1. You are provided with a STATEMENT, relevant KNOWLEDGE points, and your PREVIOUS REASONING from earlier attempts.
2. Review your PREVIOUS REASONING to understand what you have already considered and avoid repeating the same analysis.
3. Based on the KNOWLEDGE and PREVIOUS REASONING, assess the factual accuracy of the STATEMENT.
4. Before presenting your conclusion, think through the process step-by-step.
   Build upon your PREVIOUS REASONING rather than starting from scratch.
5. If the KNOWLEDGE allows you to confidently make a decision, output the final
   answer as a JSON object in the following format:
   {{
     "final_answer": "{_Factual_LABEL}" or "{_Non_Factual_LABEL}"
   }}
6. If the KNOWLEDGE is insufficient to make a judgment, issue ONE Google Search
   query that could provide additional evidence. Output the search query in JSON
   format, as follows:
   {{
     "search_query": "Your Google search query here"
   }}
7. The query should aim to obtain new information not already present in the
   KNOWLEDGE, specifically helpful for verifying the STATEMENT's accuracy.
   Refer to your PREVIOUS REASONING to avoid searching for information you have already explored.

PREVIOUS REASONING:
{_MEMORY_PLACEHOLDER}

KNOWLEDGE:
{_KNOWLEDGE_PLACEHOLDER}

STATEMENT:
{_STATEMENT_PLACEHOLDER}
"""


_MUST_HAVE_FINAL_ANSWER_WITH_MEMORY_FORMAT = f"""\
Instructions:
1. You are provided with a STATEMENT, relevant KNOWLEDGE points, and your PREVIOUS REASONING from earlier attempts.
2. Review your PREVIOUS REASONING to understand what you have already considered.
3. Based on the KNOWLEDGE and PREVIOUS REASONING, assess the factual accuracy of the STATEMENT.
4. Before presenting your final answer, think step-by-step and show your reasoning.
   Build upon your PREVIOUS REASONING rather than starting from scratch.
5. Your final answer should be either "{_Factual_LABEL}" or "{_Non_Factual_LABEL}".
6. Format your final answer as a JSON object in the following structure:
   {{
     "final_answer": "{_Factual_LABEL}" or "{_Non_Factual_LABEL}"
   }}

PREVIOUS REASONING:
{_MEMORY_PLACEHOLDER}

KNOWLEDGE:
{_KNOWLEDGE_PLACEHOLDER}

STATEMENT:
{_STATEMENT_PLACEHOLDER}
"""


_MUST_HAVE_FINAL_ANSWER_FORMAT = f"""\
Instructions:
1. You are provided with a STATEMENT and relevant KNOWLEDGE points.
2. Based on the KNOWLEDGE, assess the factual accuracy of the STATEMENT.
3. Before presenting your final answer, think step-by-step and show your reasoning. 
   Include a summary of the key points from the KNOWLEDGE as part of your reasoning.
4. Your final answer should be either "{_Factual_LABEL}" or "{_Non_Factual_LABEL}".
5. Format your final answer as a JSON object in the following structure:
   {{
     "final_answer": "{_Factual_LABEL}" or "{_Non_Factual_LABEL}"
   }}

KNOWLEDGE:
{_KNOWLEDGE_PLACEHOLDER}

STATEMENT:
{_STATEMENT_PLACEHOLDER}
"""


@dataclasses.dataclass()
class GoogleSearchResult:
    query: str
    result: str


@dataclasses.dataclass()
class FinalAnswer:
    response: str
    answer: str


def call_search(
        search_query: str,
        search_type: str = fire_config.search_type,
        num_searches: int = fire_config.num_searches,
        serper_api_key: str = shared_config.serper_api_key,
        search_postamble: str = '',  # ex: 'site:https://en.wikipedia.org'
) -> str:
    """Call Google Search to get the search result."""
    search_query += f' {search_postamble}' if search_postamble else ''

    if search_type == 'serper':
        serper_searcher = query_serper.SerperAPI(serper_api_key, k=num_searches)
        return serper_searcher.run(search_query, k=num_searches)
    else:
        raise ValueError(f'Unsupported search type: {search_type}')


def _format_memory(prior_reasonings: list[str]) -> str:
    """Format prior reasonings into a labeled, separated string."""
    return '\n---\n'.join(
        f'[Step {i+1}] {r}' for i, r in enumerate(prior_reasonings)
    )


def get_sentence_similarity(new_sent, sentences, threshold=0.9):
    if len(sentences) == 0:
        return 0
    single_embedding = sbert_model.encode(new_sent, convert_to_tensor=True).to(device)
    list_embeddings = sbert_model.encode(sentences, convert_to_tensor=True).to(device)
    similarities = util.cos_sim(single_embedding, list_embeddings)

    count_above_threshold = sum(1 for i in range(len(sentences)) if similarities[0][i].item() > threshold)
    return count_above_threshold


def final_answer_or_next_search(
        atomic_claim: str,
        past_searches: list[GoogleSearchResult],
        model: modeling.Model,
        diverse_prompt: bool = False,
        tolerance: int = 4,
        use_memory: bool = False,
        prior_reasonings: list[str] | None = None,
) -> tuple[FinalAnswer | GoogleSearchResult | None | str, dict | None, str | None]:
    """Get the next query from the model.
    atomic_claim: The claim that we need to verify.
    past_searches: The search results from the previous searches.
    model: The backbone language model we choose.
    diverse_prompt: Whether to use diverse prompt or not.
    tolerance: The number of similar queries or search results to tolerate before early stopping.
    use_memory: Whether to include prior reasoning chain-of-thought in the prompt.
    prior_reasonings: List of model responses from previous iterations.

    Returns:
        A tuple of (result, usage, model_response).
    """
    knowledge = '\n'.join([s.result for s in past_searches])
    knowledge = 'N/A' if not knowledge else knowledge

    if use_memory and prior_reasonings:
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

        if len(search_history) >= tolerance - 1 and get_sentence_similarity(search_history[-1],
                                                                            search_history[-(tolerance - 1):-1],
                                                                            threshold=0.9) >= tolerance - 2:
            full_prompt += "\n\nPlease note! We have detected multiple very similar contents in the Knowledge section. Please optimize your query so that the retrieved knowledge is as different as possible."

        if len(query_history) >= tolerance - 1 and get_sentence_similarity(query_history[-1],
                                                                           query_history[-(tolerance - 1):-1],
                                                                           threshold=0.9) >= tolerance - 2:
            full_prompt += "\nPlease note that we have detected very similar content many times in the past query history. Please pay attention to optimizing the query to make it more diverse."

    model_response, usage = model.generate(full_prompt)

    answer_or_next_query = utils.extract_json_from_output(model_response)
    if answer_or_next_query is None:
        return None, None, model_response

    if 'final_answer' in answer_or_next_query:
        final = FinalAnswer(
            response=model_response,
            answer=answer_or_next_query['final_answer'],
        )
        return final, usage, model_response

    if 'search_query' in answer_or_next_query:
        query = answer_or_next_query['search_query']
        if (len(query_history) >= (tolerance - 1)
                and get_sentence_similarity(query, query_history[-(tolerance - 1):]) >= tolerance - 1):
            return '_Early_Stop', usage, model_response
        if (len(search_history) >= tolerance
                and get_sentence_similarity(search_history[-1], search_history[-tolerance:-1]) >= tolerance - 1):
            return '_Early_Stop', usage, model_response
        search_result = GoogleSearchResult(
            query=query,
            result=call_search(query),
        )
        return search_result, usage, model_response

    print(f"Unexpected output: {answer_or_next_query}")
    return None, None, model_response


def must_get_final_answer(
        atomic_fact: str,
        searches: list[GoogleSearchResult],
        model: modeling.Model,
        use_memory: bool = False,
        prior_reasonings: list[str] | None = None,
) -> tuple[FinalAnswer | None, dict | None]:
    """Force the LLM to make a final True/False decision with whatever knowledge is available."""
    knowledge = '\n'.join(search.result for search in searches)

    if use_memory and prior_reasonings:
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
        print(f"Error in must_get_final_answer: {e}")
        return None, None


def verify_atomic_claim(
        atomic_claim: str,
        rater: modeling.Model,
        max_steps: int = fire_config.max_steps,
        max_retries: int = fire_config.max_retries,
        diverse_prompt: bool = fire_config.diverse_prompt,
        tolerance: int = fire_config.max_tolerance,
        use_memory: bool = fire_config.use_memory,
) -> tuple[FinalAnswer | None, dict[str, Any], dict | None]:
    '''
    We verify the atomic_claims by interactively calling the tools.
    :param atomic_claim: The claim that we need to verify.
    :param rater: The backbone language model we choose.
    :param max_steps: The maximum step for calling tools.
    :param max_retries: The maximum tryouts for the LLM call for each step
    :param use_memory: Whether to carry forward prior reasoning across iterations.
    :return: FinalAnswer or None, search results, usage of tokens for verifying one atomic claim.
    '''
    search_results = []
    prior_reasonings = []
    total_usage = {
        'input_tokens': 0,
        'output_tokens': 0,
    }

    stop_search = False
    for _ in range(max_steps):
        answer_or_next_search, num_tries = None, 0
        while not answer_or_next_search and num_tries <= max_retries:
            answer_or_next_search, usage, model_response = final_answer_or_next_search(
                atomic_claim, search_results, rater,
                diverse_prompt=diverse_prompt, tolerance=tolerance,
                use_memory=use_memory, prior_reasonings=prior_reasonings,
            )
            if usage is not None:
                total_usage['input_tokens'] += usage['input_tokens']
                total_usage['output_tokens'] += usage['output_tokens']
            if answer_or_next_search == '_Early_Stop':
                if use_memory and model_response:
                    prior_reasonings.append(model_response)
                stop_search = True
                break
            num_tries += 1
        if stop_search:
            break
        if answer_or_next_search is None:
            print(f'Maximum tryouts passed, still no answer or next search found.')
            break
        elif isinstance(answer_or_next_search, GoogleSearchResult):
            if use_memory and model_response:
                prior_reasonings.append(model_response)
            search_results.append(answer_or_next_search)
        elif isinstance(answer_or_next_search, FinalAnswer):
            search_dicts = {
                'google_searches': [dataclasses.asdict(s) for s in search_results]
            }
            return answer_or_next_search, search_dicts, total_usage

    # At the last step, we must reach the final answer, with whatever the information we have so far.
    final_answer, num_tries = None, 0
    while not final_answer and num_tries <= max_retries:
        num_tries += 1
        final_answer, usage = must_get_final_answer(
            atomic_claim, searches=search_results, model=rater,
            use_memory=use_memory, prior_reasonings=prior_reasonings,
        )
        if usage is not None:
            total_usage['input_tokens'] += usage['input_tokens']
            total_usage['output_tokens'] += usage['output_tokens']
    search_dicts = {
        'google_searches': [dataclasses.asdict(s) for s in search_results]
    }
    return final_answer, search_dicts, total_usage

if __name__ == '__main__':
    pass