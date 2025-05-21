import os
import yaml
import logging
import traceback
import shutil
import argparse
from langchain_community.document_loaders import PyPDFDirectoryLoader


# load parameters from "params.yaml"
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

LOG_PATH = params["LOG_PATH"]
GAME_RULES_DATA_PATH = params["GAME_RULES_DATA_PATH"]
CHROMA_DB_PATH = params["CHROMA_DB_PATH"]
CHROMA_DB_FILE = params["CHROMA_DB_FILE"]
CHUNK_SIZE = params["CHUNK_SIZE"]
CHUNK_OVERLAP = params["CHUNK_OVERLAP"]

# setup logging
LOG_DIR = os.path.join(os.getcwd(), LOG_PATH)
os.makedirs(LOG_DIR, exist_ok=True)  # Create the logs directory if it doesn't exist
LOG_FILE = os.path.join(LOG_DIR, "01_populate_db.log")

def setup_logger(name, log_file):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers on re-imports
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


# load game rules data
def load_docs(logger):
    try:
        doc_loader = PyPDFDirectoryLoader(GAME_RULES_DATA_PATH, glob="*.pdf")   
        logger.info(f"Loading documents from {GAME_RULES_DATA_PATH}")
        return doc_loader.load()
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        logger.debug(traceback.format_exc())
        return []


def main():
    logger = setup_logger("create_db_logger", LOG_FILE)
    logger.info("*******************Starting db population*******************")
    
    logger.info("*******************Loading docs from the directory*******************")
    documents = load_docs(logger)
    logger.info(f"Loaded {len(documents)} documents")
    if not documents:
        logger.error("No documents loaded. Exiting.")
        return
    logger.info(f"First document: {documents[0]}")
    logger.info("*******************Docs loaded successfully*******************")
    
    logger.info("*******************DB population completed*******************")


if __name__ == "__main__":
    main()