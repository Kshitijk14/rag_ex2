import os
import re
import string
import math
import pandas as pd
from datetime import datetime
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set


def normalize_text(text: str) -> str:
    """
    Normalize text for evaluation by:
    - Converting to lowercase
    - Removing punctuation
    - Removing extra whitespace
    """
    # to lowercase
    text = text.lower()
    
    # - punctuations
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # - extra whitespaces and splitting → tokens
    tokens = text.split()
    
    # joining back with single spaces
    return ' '.join(tokens)

def get_tokens(text: str) -> List[str]:
    """
    Tokenize normalized text into individual words
    """
    normalized = normalize_text(text)
    return normalized.split()


def calc_f1_score(predicted: str, ground_truth: str) -> Dict[str, float]:
    """
    Calculate F1 score between predicted and ground truth text
    
    Args:
        predicted: Generated answer from RAG system
        ground_truth: Expected/reference answer
    
    Returns:
        Dictionary containing precision, recall, and f1 scores
    """
    # get tokens (both texts)
    pred_tokens = get_tokens(predicted)
    truth_tokens = get_tokens(ground_truth)
    
    # convert → Counter objects (easier intersection calc)
    pred_counter = Counter(pred_tokens)
    truth_counter = Counter(truth_tokens)
    
    # calc intersection (common tokens)
    common_tokens = pred_counter & truth_counter
    num_common = sum(common_tokens.values())
    
    # calc P and R
    if len(pred_tokens) == 0:
        precision = 0.0
    else:
        precision = num_common / len(pred_tokens)
    
    if len(truth_tokens) == 0:
        recall = 0.0
    else:
        recall = num_common / len(truth_tokens)
    
    # calc F1
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def calc_em_score(predicted: str, ground_truth: str) -> Dict[str, float]:
    """
    Calculate Exact Match score between predicted and ground truth text
    
    Args:
        predicted: Generated answer from RAG system
        ground_truth: Expected/reference answer
    
    Returns:
        Dictionary containing exact match score (0 or 1)
    """
    # Normalize both texts
    pred_normalized = normalize_text(predicted)
    truth_normalized = normalize_text(ground_truth)
    
    # Check if they're exactly the same
    exact_match = 1.0 if pred_normalized == truth_normalized else 0.0
    
    return {
        'exact_match': exact_match
    }

def calc_rouge_l_score(predicted: str, ground_truth: str) -> Dict[str, float]:
    """
    Calculate ROUGE-L score based on Longest Common Subsequence (LCS)
    
    Args:
        predicted: Generated answer from RAG system
        ground_truth: Expected/reference answer
    
    Returns:
        Dictionary containing ROUGE-L precision, recall, and F1 scores
    """
    def lcs_length(x: List[str], y: List[str]) -> int:
        """Calculate length of Longest Common Subsequence"""
        m, n = len(x), len(y)
        
        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Fill the DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    # Get tokens for both texts
    pred_tokens = get_tokens(predicted)
    truth_tokens = get_tokens(ground_truth)
    
    if len(pred_tokens) == 0 and len(truth_tokens) == 0:
        return {'rouge_l_precision': 1.0, 'rouge_l_recall': 1.0, 'rouge_l_f1': 1.0}
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return {'rouge_l_precision': 0.0, 'rouge_l_recall': 0.0, 'rouge_l_f1': 0.0}
    
    # Calculate LCS length
    lcs_len = lcs_length(pred_tokens, truth_tokens)
    
    # Calculate precision and recall
    precision = lcs_len / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
    recall = lcs_len / len(truth_tokens) if len(truth_tokens) > 0 else 0.0
    
    # Calculate F1 score
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return {
        'rouge_l_precision': precision,
        'rouge_l_recall': recall,
        'rouge_l_f1': f1
    }

def calc_bleu_score(predicted: str, ground_truth: str, max_n: int = 4) -> Dict[str, float]:
    """
    Calculate BLEU score between predicted and ground truth text
    
    Args:
        predicted: Generated answer from RAG system
        ground_truth: Expected/reference answer
        max_n: Maximum n-gram size to consider (default: 4)
    
    Returns:
        Dictionary containing BLEU scores for different n-grams and overall BLEU
    """
    def get_ngrams(tokens: List[str], n: int) -> Counter:
        """Get n-grams from tokens"""
        if len(tokens) < n:
            return Counter()
        return Counter([tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)])
    
    def calculate_brevity_penalty(pred_len: int, ref_len: int) -> float:
        """Calculate brevity penalty for BLEU score"""
        if pred_len > ref_len:
            return 1.0
        elif pred_len == 0:
            return 0.0
        else:
            return math.exp(1 - ref_len / pred_len)
    
    # Get tokens
    pred_tokens = get_tokens(predicted)
    ref_tokens = get_tokens(ground_truth)
    
    if len(pred_tokens) == 0:
        return {f'bleu_{i}': 0.0 for i in range(1, max_n + 1)} | {'bleu': 0.0}
    
    # Calculate precision for each n-gram
    precisions = []
    bleu_scores = {}
    
    for n in range(1, max_n + 1):
        pred_ngrams = get_ngrams(pred_tokens, n)
        ref_ngrams = get_ngrams(ref_tokens, n)
        
        if len(pred_ngrams) == 0:
            precision = 0.0
        else:
            # Count matches
            matches = 0
            for ngram in pred_ngrams:
                matches += min(pred_ngrams[ngram], ref_ngrams.get(ngram, 0))
            
            precision = matches / len(pred_ngrams)
        
        precisions.append(precision)
        bleu_scores[f'bleu_{n}'] = precision
    
    # Calculate overall BLEU score (geometric mean of precisions * brevity penalty)
    if all(p > 0 for p in precisions):
        geometric_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    else:
        geometric_mean = 0.0
    
    brevity_penalty = calculate_brevity_penalty(len(pred_tokens), len(ref_tokens))
    overall_bleu = geometric_mean * brevity_penalty
    
    bleu_scores['bleu'] = overall_bleu
    bleu_scores['brevity_penalty'] = brevity_penalty
    
    return bleu_scores

def calc_faithfulness_score(predicted: str, context: str) -> Dict[str, float]:
    """
    Calculate faithfulness score - how much of the predicted answer is supported by context
    
    Args:
        predicted: Generated answer from RAG system
        context: Retrieved context used for generation
    
    Returns:
        Dictionary containing faithfulness percentage
    """
    # Get tokens
    pred_tokens = get_tokens(predicted)
    context_tokens = get_tokens(context)
    
    if len(pred_tokens) == 0:
        return {'faithfulness': 1.0}  # Empty prediction is perfectly faithful
    
    if len(context_tokens) == 0:
        return {'faithfulness': 0.0}  # No context means no faithfulness
    
    # Count how many predicted tokens appear in context
    context_token_set = set(context_tokens)
    supported_tokens = 0
    
    for token in pred_tokens:
        if token in context_token_set:
            supported_tokens += 1
    
    faithfulness = supported_tokens / len(pred_tokens)
    
    return {
        'faithfulness': faithfulness
    }

def calc_all_generation_scores(predicted: str, ground_truth: str, context: str = "") -> Dict[str, float]:
    """
    Calculate all generation evaluation scores
    
    Args:
        predicted: Generated answer from RAG system
        ground_truth: Expected/reference answer
        context: Retrieved context (for faithfulness calculation)
    
    Returns:
        Dictionary containing all scores
    """
    scores = {}
    
    # F1 Score
    scores.update(calc_f1_score(predicted, ground_truth))
    
    # Exact Match
    scores.update(calc_em_score(predicted, ground_truth))
    
    # ROUGE-L
    scores.update(calc_rouge_l_score(predicted, ground_truth))
    
    # BLEU Score
    scores.update(calc_bleu_score(predicted, ground_truth))
    
    # Faithfulness (only if context is provided)
    if context:
        scores.update(calc_faithfulness_score(predicted, context))
    
    return scores


def calc_mrr_score(retrieved_doc_ids: List[str], relevant_doc_ids: List[str]) -> Dict[str, float]:
    """
    Calculate Mean Reciprocal Rank (MRR)
    
    Args:
        retrieved_doc_ids: List of document IDs in retrieval order
        relevant_doc_ids: List of relevant document IDs (ground truth)
    
    Returns:
        Dictionary containing MRR score
    """
    if not retrieved_doc_ids or not relevant_doc_ids:
        return {'mrr': 0.0}
    
    # Convert to sets for faster lookup
    relevant_set = set(relevant_doc_ids)
    
    # Find the rank of the first relevant document
    for rank, doc_id in enumerate(retrieved_doc_ids, 1):
        if doc_id in relevant_set:
            return {'mrr': 1.0 / rank}
    
    # No relevant document found
    return {'mrr': 0.0}


def calc_ndcg_at_k_score(retrieved_doc_ids: List[str], relevant_doc_ids: List[str], k: int = 5) -> Dict[str, float]:
    """
    Calculate Normalized Discounted Cumulative Gain (nDCG@k)
    
    Args:
        retrieved_doc_ids: List of document IDs in retrieval order
        relevant_doc_ids: List of relevant document IDs (ground truth)
        k: Number of top documents to consider
    
    Returns:
        Dictionary containing nDCG@k score
    """
    if not retrieved_doc_ids or not relevant_doc_ids:
        return {f'ndcg_at_{k}': 0.0}
    
    # Convert to set for faster lookup
    relevant_set = set(relevant_doc_ids)
    
    # Calculate DCG (Discounted Cumulative Gain)
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_doc_ids[:k]):
        if doc_id in relevant_set:
            # Binary relevance: relevant=1, non-relevant=0
            # DCG formula: sum(rel_i / log2(i + 2)) for i from 0 to k-1
            dcg += 1.0 / math.log2(i + 2)
    
    # Calculate IDCG (Ideal DCG) - best possible DCG
    # This is DCG when all relevant docs are at the top
    num_relevant = min(len(relevant_doc_ids), k)
    idcg = 0.0
    for i in range(num_relevant):
        idcg += 1.0 / math.log2(i + 2)
    
    # Calculate nDCG
    if idcg == 0:
        ndcg = 0.0
    else:
        ndcg = dcg / idcg
    
    return {f'ndcg_at_{k}': ndcg}


def calc_precision_at_k_score(retrieved_doc_ids: List[str], relevant_doc_ids: List[str], k: int = 5) -> Dict[str, float]:
    """
    Calculate Precision@k
    
    Args:
        retrieved_doc_ids: List of document IDs in retrieval order
        relevant_doc_ids: List of relevant document IDs (ground truth)
        k: Number of top documents to consider
    
    Returns:
        Dictionary containing Precision@k score
    """
    if not retrieved_doc_ids or not relevant_doc_ids:
        return {f'precision_at_{k}': 0.0}
    
    # Convert to set for faster lookup
    relevant_set = set(relevant_doc_ids)
    
    # Count relevant documents in top-k retrieved documents
    top_k_retrieved = retrieved_doc_ids[:k]
    relevant_retrieved = sum(1 for doc_id in top_k_retrieved if doc_id in relevant_set)
    
    # Precision@k = (relevant docs in top-k) / k
    precision_at_k = relevant_retrieved / k
    
    return {f'precision_at_{k}': precision_at_k}


def calc_recall_at_k_score(retrieved_doc_ids: List[str], relevant_doc_ids: List[str], k: int = 5) -> Dict[str, float]:
    """
    Calculate Recall@k
    
    Args:
        retrieved_doc_ids: List of document IDs in retrieval order
        relevant_doc_ids: List of relevant document IDs (ground truth)
        k: Number of top documents to consider
    
    Returns:
        Dictionary containing Recall@k score
    """
    if not retrieved_doc_ids or not relevant_doc_ids:
        return {f'recall_at_{k}': 0.0}
    
    # Convert to set for faster lookup
    relevant_set = set(relevant_doc_ids)
    
    # Count relevant documents in top-k retrieved documents
    top_k_retrieved = retrieved_doc_ids[:k]
    relevant_retrieved = sum(1 for doc_id in top_k_retrieved if doc_id in relevant_set)
    
    # Recall@k = (relevant docs in top-k) / (total relevant docs)
    recall_at_k = relevant_retrieved / len(relevant_doc_ids)
    
    return {f'recall_at_{k}': recall_at_k}


def calc_all_retrieval_scores(retrieved_doc_ids: List[str], relevant_doc_ids: List[str], k: int = 5) -> Dict[str, float]:
    """
    Calculate all retrieval evaluation scores
    
    Args:
        retrieved_doc_ids: List of document IDs in retrieval order
        relevant_doc_ids: List of relevant document IDs (ground truth)
        k: Number of top documents to consider for @k metrics
    
    Returns:
        Dictionary containing all retrieval scores
    """
    scores = {}
    
    # MRR
    scores.update(calc_mrr_score(retrieved_doc_ids, relevant_doc_ids))
    
    # nDCG@k
    scores.update(calc_ndcg_at_k_score(retrieved_doc_ids, relevant_doc_ids, k))
    
    # Precision@k
    scores.update(calc_precision_at_k_score(retrieved_doc_ids, relevant_doc_ids, k))
    
    # Recall@k
    scores.update(calc_recall_at_k_score(retrieved_doc_ids, relevant_doc_ids, k))
    
    return scores


# # test the utils
# if __name__ == "__main__":
#     # Test all metrics
#     predicted = "The king can move one square in any direction horizontally, vertically, or diagonally."
#     ground_truth = "A king moves one square in any direction: horizontally, vertically, or diagonally."
#     context = "Chess piece movement rules: The king is the most important piece. A king moves one square in any direction - horizontally, vertically, or diagonally. The king cannot move into check."
#     retrieved_doc_ids = [
#         "doc_7", "doc_3", "doc_5", "doc_2", "doc_1"
#     ]
#     relevant_doc_ids = [
#         "doc_2", "doc_5", "doc_9"
#     ]
#     k = 5
    
#     print("=== Testing All Generation Metrics ===")
#     print(f"Predicted: {predicted}")
#     print(f"Ground Truth: {ground_truth}")
#     print(f"Context: {context[:100]}...")
#     print(f"Retrieved Docs: {retrieved_doc_ids}")
#     print(f"Relevant Docs: {relevant_doc_ids}")
#     print()
    
#     generation_scores = calc_all_generation_scores(predicted, ground_truth, context)
#     for metric, score in generation_scores.items():
#         print(f"{metric}: {score:.3f}")
    
#     retrieval_scores = calc_all_retrieval_scores(retrieved_doc_ids, relevant_doc_ids, k=k)
#     for metric, score in retrieval_scores.items():
#         print(f"{metric}: {score:.3f}")