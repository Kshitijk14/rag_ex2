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
from utils import calc_f1_score


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
        predicted_answer = query_rag(question, logger)
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000  # Convert to milliseconds
        
        # Calculate F1 score
        f1_metrics = calc_f1_score(predicted_answer, ground_truth)
        
        return {
            'question': question,
            'predicted_answer': predicted_answer,
            'ground_truth': ground_truth,
            'latency_ms': latency_ms,
            'precision': f1_metrics['precision'],
            'recall': f1_metrics['recall'],
            'f1_score': f1_metrics['f1'],
            'success': True,
            'error': None
        }
        
    except Exception as e:
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        return {
            'question': question,
            'predicted_answer': '',
            'ground_truth': ground_truth,
            'latency_ms': latency_ms,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'success': False,
            'error': str(e)
        }

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
        logger.info(f"Loading evaluation data from {path}")
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

    logger.info(f"Total combined evaluation questions: {len(eval_data)}")
    
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
        logger.info(f"F1 Score: {result['f1_score']:.3f}, Latency: {result['latency_ms']:.1f}ms")
    
    # Calculate aggregate metrics
    successful_results = [r for r in results if r['success']]
    if successful_results:
        avg_f1 = sum(r['f1_score'] for r in successful_results) / len(successful_results)
        avg_precision = sum(r['precision'] for r in successful_results) / len(successful_results)
        avg_recall = sum(r['recall'] for r in successful_results) / len(successful_results)
        avg_latency = sum(r['latency_ms'] for r in successful_results) / len(successful_results)
        
        logger.info("*******************Evaluation Summary*******************")
        logger.info(f"Total Questions: {len(eval_data)}")
        logger.info(f"Successful Evaluations: {len(successful_results)}")
        logger.info(f"Average F1 Score: {avg_f1:.3f}")
        logger.info(f"Average Precision: {avg_precision:.3f}")
        logger.info(f"Average Recall: {avg_recall:.3f}")
        logger.info(f"Average Latency: {avg_latency:.1f}ms")
    
    # Save results to CSV
    save_results_to_csv(results, logger)
    
    logger.info("*******************[Pipeline 3] RAG Evaluation Completed*******************")
    return results

if __name__ == "__main__":
    run_evaluation()