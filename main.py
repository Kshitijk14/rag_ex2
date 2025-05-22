import os
import yaml
import traceback
import argparse
from logger import setup_logger
from pipeline.stage_01_populate_db import run_populate_db
from pipeline.stage_02_query_data import run_query_rag

# load parameters from "params.yaml"
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

LOG_PATH = params["LOG_PATH"]

# setup logging
LOG_DIR = os.path.join(os.getcwd(), LOG_PATH)
os.makedirs(LOG_DIR, exist_ok=True)  # Create the logs directory if it doesn't exist
LOG_FILE = os.path.join(LOG_DIR, "main.log")

def main():
    parser = argparse.ArgumentParser(description="Main pipeline for Chroma DB and RAG")
    parser.add_argument("--reset", action="store_true", help="Reset the Chroma DB")
    args = parser.parse_args()
    
    try:
        logger = setup_logger("main_logger", LOG_FILE)
        logger.info(" ")
        logger.info("*******************Main pipeline started*******************")
        # Step 1: Populate the Chroma DB
        print("Running DB population (with reset)...")
        logger.info("*******************Populating Chroma DB started (with reset)*******************")
        run_populate_db(reset=args.reset)
    except Exception as e:
        logger.error(f"Error in populating DB: {e}")
        logger.debug(traceback.format_exc())
        return
    
    try:
        # Step 2: Query the RAG pipeline
        print("\nQuerying the DB...")
        logger.info("*******************Querying Chroma DB started*******************")
        query = input("Enter your query: ")
    except Exception as e:
        logger.error(f"Error in querying DB: {e}")
        logger.debug(traceback.format_exc())
        return
    
    try:
        # Step 3: Run the query through the RAG pipeline
        print("\nRunning query...")
        logger.info("*******************Running query through RAG pipeline started*******************")
        response = run_query_rag(query)
        print("\nLLM Response:")
        logger.info("*******************LLM Response*******************")
        print(response)
        logger.info(response)
        logger.info("*******************Main pipeline finished*******************")
    except Exception as e:
        logger.error(f"Error in querying RAG pipeline: {e}")
        logger.debug(traceback.format_exc())
        return

if __name__ == "__main__":
    main()
