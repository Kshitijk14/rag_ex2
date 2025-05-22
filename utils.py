import os
import re
import string
from collections import Counter
from typing import List, Dict, Tuple
import pandas as pd
from datetime import datetime


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

# # test the utils
# if __name__ == "__main__":
#     # test the F1 calc
#     predicted = "The king can move one square in any direction horizontally, vertically, or diagonally."
#     ground_truth = "A king moves one square in any direction: horizontally, vertically, or diagonally."
    
#     scores = calc_f1_score(predicted, ground_truth)
    
#     print("F1 Score Evaluation:")
#     print(f"Predicted: {predicted}")
#     print(f"Ground Truth: {ground_truth}")
#     print(f"Precision: {scores['precision']:.3f}")
#     print(f"Recall: {scores['recall']:.3f}")
#     print(f"F1 Score: {scores['f1']:.3f}")