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
EVALUATION_DATA_PATH = params.get("EVALUATION_DATA_PATH", "chess_questions.json")
RESULTS_CSV_PATH = params.get("RESULTS_CSV_PATH", "chess_evaluation_results.csv")

# Setup logging
LOG_DIR = os.path.join(os.getcwd(), LOG_PATH)
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "stage_03_evaluate.log")