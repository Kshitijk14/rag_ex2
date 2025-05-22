import os
import json
import yaml
import traceback
import time
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
from logger import setup_logger
from pipeline.stage_02_query_data import query_rag
from utils import calc_all_generation_scores


# Load parameters from params.yaml
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

LOG_PATH = params["LOG_PATH"]
GENERATION_MODEL = params["GENERATION_MODEL"]
EVALUATION_DATA_PATHS = params.get("EVALUATION_DATA_PATHS", [])
RESULTS_CSV_PATH = params.get("RESULTS_CSV_PATH", "evaluation_results.csv")

# Setup logging
LOG_DIR = os.path.join(os.getcwd(), LOG_PATH)
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "stage_03_evaluate_rag.log")


def load_evaluation_data(file_path: str) -> List[Dict]:
    """
    Load evaluation questions and ground truth answers
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Evaluation data file not found: {file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file: {e}")
        return []

def evaluate_single_query(question: str, ground_truth: str, logger) -> Dict[str, Any]:
    """
    Evaluate a single query through the RAG pipeline
    
    Returns:
        Dictionary containing metrics and metadata
    """
    # Measure latency
    start_time = time.time()
    
    # Get RAG response
    try:
        # Get both answer and context
        rag_result = query_rag(question, logger)
        predicted_answer = rag_result['answer']
        context = rag_result['context']
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Calculate all generation scores
        all_scores = calc_all_generation_scores(predicted_answer, ground_truth, context)
        
        result = {
            'question': question,
            'predicted_answer': predicted_answer,
            'ground_truth': ground_truth,
            'context': context[:200] + "..." if len(context) > 200 else context,  # Truncate for CSV
            'latency_ms': latency_ms,
            'success': True,
            'error': None
        }
        
        # Add all metric scores to result
        result.update(all_scores)
        
        return result
        
    except Exception as e:
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        result = {
            'question': question,
            'predicted_answer': '',
            'ground_truth': ground_truth,
            'latency_ms': latency_ms,
            'success': False,
            'error': str(e)
        }
        
        # Add zero scores for all metrics
        zero_scores = {
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
            'exact_match': 0.0,
            'rouge_l_precision': 0.0, 'rouge_l_recall': 0.0, 'rouge_l_f1': 0.0,
            'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0, 'bleu': 0.0,
            'faithfulness': 0.0
        }
        result.update(zero_scores)
        
        return result

def save_results_to_csv(results: List[Dict], logger):
    """
    Save evaluation results to CSV file
    """
    try:
        df = pd.DataFrame(results)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(RESULTS_CSV_PATH), exist_ok=True)
        
        # Check if CSV exists to append or create new
        if os.path.exists(RESULTS_CSV_PATH):
            # Append to existing CSV
            existing_df = pd.read_csv(RESULTS_CSV_PATH)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.to_csv(RESULTS_CSV_PATH, index=False)
            logger.info(f"Appended {len(results)} results to existing CSV: {RESULTS_CSV_PATH}")
        else:
            # Create new CSV
            df.to_csv(RESULTS_CSV_PATH, index=False)
            logger.info(f"Created new CSV with {len(results)} results: {RESULTS_CSV_PATH}")
            
    except Exception as e:
        logger.error(f"Error saving results to CSV: {e}")
        logger.debug(traceback.format_exc())


def run_evaluation():
    """
    Main evaluation function
    """
    logger = setup_logger("evaluation_logger", LOG_FILE)
    logger.info(" ")
    logger.info("*******************[Pipeline 3] Starting RAG Evaluation*******************")
    
    # # Load evaluation data
    # eval_data = load_evaluation_data(EVALUATION_DATA_PATH)
    # if not eval_data:
    #     logger.error("No evaluation data found. Exiting.")
    #     return
    
    # logger.info(f"Loaded {len(eval_data)} evaluation questions")
    
    # Load and combine data from all paths
    eval_data = []
    for path in EVALUATION_DATA_PATHS:
        logger.info(f"[Stage 1] Loading evaluation data from {path}")
        data = load_evaluation_data(path)
        if not data:
            logger.warning(f"No valid data found in {path}")
            continue
        for item in data:
            item['source_file'] = os.path.basename(path)  # Optional: track source
        eval_data.extend(data)

    if not eval_data:
        logger.error("No evaluation data found from any source. Exiting.")
        return

    logger.info(f"[Stage 2] Total combined evaluation questions: {len(eval_data)}")
    
    # Run evaluation for each question
    results = []
    for i, item in enumerate(eval_data):
        logger.info(f"Evaluating question {i+1}/{len(eval_data)}: {item['question'][:50]}...")
        
        result = evaluate_single_query(
            question=item['question'],
            ground_truth=item['ground_truth'],
            logger=logger
        )
        
        # Add metadata
        result.update({
            'timestamp': datetime.now().isoformat(),
            'model_name': GENERATION_MODEL,
            'question_id': i,
            'category': item.get('category', 'unknown'),
            'difficulty': item.get('difficulty', 'unknown'),
            'source_file': item.get('source_file', 'unknown'),
        })
        
        results.append(result)
        
        # Log key metrics for this question
        logger.info(f"[Stage 3] Scores -> "
                    f"F1: {result.get('f1', 0):.3f}, "
                    f"Precision: {result.get('precision', 0):.3f}, "
                    f"Recall: {result.get('recall', 0):.3f}, "
                    f"EM: {result.get('exact_match', 0):.3f}, "
                    f"ROUGE-L: {result.get('rouge_l_f1', 0):.3f}, "
                    f"BLEU: {result.get('bleu', 0):.3f}, "
                    f"BLEU-1: {result.get('bleu_1', 0):.3f}, "
                    f"BLEU-2: {result.get('bleu_2', 0):.3f}, "
                    f"BLEU-3: {result.get('bleu_3', 0):.3f}, "
                    f"BLEU-4: {result.get('bleu_4', 0):.3f}, "
                    f"Faithfulness: {result.get('faithfulness', 0):.3f}, "
                    f"Latency: {result['latency_ms']:.1f}ms")
    
    # Calculate aggregate metrics
    successful_results = [r for r in results if r['success']]
    if successful_results:
        # Calculate averages for all metrics
        metrics_to_average = [
            'f1', 'precision', 'recall', 'exact_match',
            'rouge_l_f1', 'rouge_l_precision', 'rouge_l_recall',
            'bleu', 'bleu_1', 'bleu_2', 'bleu_3', 'bleu_4',
            'faithfulness', 'latency_ms'
        ]
        
        avg_metrics = {}
        for metric in metrics_to_average:
            if metric in successful_results[0]:  # Check if metric exists
                avg_metrics[f'avg_{metric}'] = sum(r[metric] for r in successful_results) / len(successful_results)
        
        logger.info("*******************Evaluation Summary*******************")
        logger.info(f"Total Questions: {len(eval_data)}")
        logger.info(f"Successful Evaluations: {len(successful_results)}")
        logger.info(f"Average F1 Score: {avg_metrics.get('avg_f1', 0):.3f}")
        logger.info(f"Average Precision: {avg_metrics.get('avg_precision', 0):.3f}")
        logger.info(f"Average Recall: {avg_metrics.get('avg_recall', 0):.3f}")
        logger.info(f"Average Exact Match: {avg_metrics.get('avg_exact_match', 0):.3f}")
        logger.info(f"Average ROUGE-L Precision: {avg_metrics.get('avg_rouge_l_precision', 0):.3f}")
        logger.info(f"Average ROUGE-L Recall: {avg_metrics.get('avg_rouge_l_recall', 0):.3f}")
        logger.info(f"Average ROUGE-L F1: {avg_metrics.get('avg_rouge_l_f1', 0):.3f}")
        logger.info(f"Average BLEU Score: {avg_metrics.get('avg_bleu', 0):.3f}")
        logger.info(f"Average BLEU-1: {avg_metrics.get('avg_bleu_1', 0):.3f}")
        logger.info(f"Average BLEU-2: {avg_metrics.get('avg_bleu_2', 0):.3f}")
        logger.info(f"Average BLEU-3: {avg_metrics.get('avg_bleu_3', 0):.3f}")
        logger.info(f"Average BLEU-4: {avg_metrics.get('avg_bleu_4', 0):.3f}")
        logger.info(f"Average Faithfulness: {avg_metrics.get('avg_faithfulness', 0):.3f}")
        logger.info(f"Average Latency: {avg_metrics.get('avg_latency_ms', 0):.1f}ms")
    
    # Save results to CSV
    save_results_to_csv(results, logger)
    
    logger.info("*******************[Pipeline 3] RAG Evaluation Completed*******************")
    return results

if __name__ == "__main__":
    run_evaluation()