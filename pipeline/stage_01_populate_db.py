import os
import yaml
import traceback
import shutil
import argparse
from tqdm import tqdm
from logger import setup_logger
# from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import PyMuPDFLoader # better layout accuracy; handles unicode, tables, & symbols better; slightly slower, but still faster than PDFMiner; supports text extraction and images 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from get_embedding_func import embedding_function
from langchain_chroma import Chroma


# load parameters from "params.yaml"
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

LOG_PATH = params["LOG_PATH"]
GAME_RULES_DATA_PATH = params["GAME_RULES_DATA_PATH"]
CHROMA_DB_PATH = params["CHROMA_DB_PATH"]
CHROMA_DB_FILE = params["CHROMA_DB_FILE"]
CHUNK_SIZE = params["CHUNK_SIZE"]
CHUNK_OVERLAP = params["CHUNK_OVERLAP"]
BATCH_SIZE = params["BATCH_SIZE"]

# setup logging
LOG_DIR = os.path.join(os.getcwd(), LOG_PATH)
os.makedirs(LOG_DIR, exist_ok=True)  # Create the logs directory if it doesn't exist
LOG_FILE = os.path.join(LOG_DIR, "stage_01_populate_db.log")


def load_docs(logger):
    try:
        logger.info("*******************Loading docs from the directory*******************")
        
        # doc_loader = PyPDFDirectoryLoader(GAME_RULES_DATA_PATH, glob="*.pdf")   
        # logger.info(f"[Stage 1] Loading documents from {GAME_RULES_DATA_PATH}")
        
        all_docs = []
        pdf_files = []
        logger.info(f"[Stage 1] Scanning directory: {GAME_RULES_DATA_PATH}")
        
        # Collect all PDF paths (including subfolders)
        for root, dirs, files in os.walk(GAME_RULES_DATA_PATH):
            for file in files:
                if file.lower().endswith(".pdf"):
                    pdf_files.append(os.path.join(root, file))
                    logger.info(f"[Stage 2] Loading: {pdf_files}")
                    
                    for file_path in tqdm(pdf_files, desc="Loading PDFs"):
                        try:
                            loader = PyMuPDFLoader(file_path)
                            docs = loader.load()
                            all_docs.extend(docs)
                        except Exception as inner_e:
                            logger.warning(f"Failed to load {file_path}: {inner_e}")
        
        logger.info(f"[Stage 3] Loaded {len(all_docs)} documents.")
        logger.info("*******************Docs loaded successfully*******************")
        
        # return doc_loader.load()
        return all_docs
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        logger.debug(traceback.format_exc())
        return []

def split_docs(docs: list[Document], logger):
    try:
        logger.info("*******************Splitting docs into chunks*******************")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            is_separator_regex=False,
        )
        logger.info("[Stage 2] Splitting documents into chunks")
        
        logger.info("*******************Chunks created successfully*******************")
        return text_splitter.split_documents(docs)
    except Exception as e:
        logger.error(f"Error splitting documents: {e}")
        logger.debug(traceback.format_exc())
        return []

def process_in_batches(documents, batch_size, ingest_fn, logger):
    """
    Process documents in batches through a given ingestion function.
    """
    total = len(documents)
    logger.info(f"Processing {total} documents in batches of {batch_size}...")

    for i in tqdm(range(0, total, batch_size), desc="Ingesting batches"):
        batch = documents[i:i + batch_size]
        try:
            ingest_fn(batch)
        except Exception as e:
            logger.error(f"Failed to ingest batch {i // batch_size}: {e}")
            logger.debug(traceback.format_exc())

# logic to add & update items in the existing db
def save_to_chroma_db(chunks: list[Document], logger):
    try:
        logger.info("*******************Saving chunks to Chroma DB*******************")
        
        # load the existing db
        db = Chroma(
            embedding_function=embedding_function(),
            persist_directory=CHROMA_DB_PATH,
        )
        logger.info(f"[Stage 3.1] Loading existing DB from path: {CHROMA_DB_PATH}")
        
        # calculate "page:chunk" IDs
        # Step 1: Assign chunk IDs
        chunks_with_ids = calc_chunk_ids(chunks)
        logger.info(f"[Stage 3.2] Calculated chunk IDs for {len(chunks_with_ids)} chunks")
        
        # add/update the docs
        # Step 2: Get existing IDs from the DB
        existing_items = db.get(include=[]) # IDs are always included by default
        existing_ids = set(existing_items["ids"])
        logger.info(f"[Stage 3.3] No. of existing items (i.e. docs) in the db: {len(existing_ids)}")
        
        # only add docs that don't exist in the db
        # Step 3: Filter only new chunks
        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["chunk_id"] not in existing_ids:
                new_chunks.append(chunk)
        logger.info(f"[Stage 3.4] No. of new chunks to add: {len(new_chunks)}")
        
        # ✅ Step 4: Safety Net — remove duplicates within `new_chunks`
        seen_ids = set()
        unique_new_chunks = []
        for chunk in new_chunks:
            cid = chunk.metadata["chunk_id"]
            if cid not in seen_ids:
                seen_ids.add(cid)
                unique_new_chunks.append(chunk)
        
        logger.info(f"[Stage 3.5] Unique new chunks after deduplication: {len(unique_new_chunks)}")
        
        # if len(new_chunks):
        #     logger.info(f"[Stage 3.5(a)] Adding {len(new_chunks)} new chunks to the db")
        #     new_chunk_ids = [chunk.metadata["chunk_id"] for chunk in new_chunks]
        #     db.add_documents(new_chunks, ids=new_chunk_ids)
        # else:
        #     logger.info("[Stage 3.5(b)] No new chunks to add to the db")
        
        # Step 5: Ingest in batches
        if unique_new_chunks:
            logger.info("[Stage 3.6(a)] Ingesting new chunks in batches...")
            process_in_batches(
                documents=unique_new_chunks,
                batch_size=BATCH_SIZE,
                ingest_fn=lambda batch: db.add_documents(batch, ids=[doc.metadata["chunk_id"] for doc in batch]),
                logger=logger
            )
        else:
            logger.info("[Stage 3.6(b)] No new unique chunks to add to DB")
        
        logger.info("*******************Chunks saved to Chroma DB successfully*******************")        
    except Exception as e:
        logger.error(f"Error saving to Chroma DB: {e}")
        logger.debug(traceback.format_exc())
        return []

def calc_chunk_ids(chunks):
    # Page Source : Page Number : Chunk Index
    last_page_id = None
    curr_chunk_idx = 0
    
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        
        # Normalize and standardize the path for cross-platform consistency
        norm_source = os.path.normpath(source)
        rel_source = os.path.relpath(norm_source, GAME_RULES_DATA_PATH).replace("\\", "/")
        
        curr_page_id = (f"{rel_source}:{page}")
        
        # if the page ID is the same as the last one, increment the index
        if curr_page_id == last_page_id:
            curr_chunk_idx += 1
        else:
            curr_chunk_idx = 0
        
        # calculate the new chunk ID
        chunk_id = (f"{curr_page_id}:{curr_chunk_idx}")
        last_page_id = curr_page_id
        
        # add the chunk ID to the metadata
        chunk.metadata["chunk_id"] = chunk_id
    
    return chunks

def clear_database():
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)


def run_populate_db(reset=False):
    try:
        logger = setup_logger("populate_db_logger", LOG_FILE)
        logger.info(" ")
        logger.info("*******************[Pipeline 1] Starting db population*******************")
        
        # check if the db should be cleared (using the --clear flag)
        if reset:
            logger.info("Clearing the database...")
            clear_database()
        
        # create (or update) the db
        documents = load_docs(logger)
        # logger.info(f"Loaded {len(documents)} documents")
        if not documents:
            logger.error("No documents loaded. Exiting.")
            return
        # logger.info(f"First document: {documents[0]}")
        
        chunks = split_docs(documents, logger)
        # logger.info(f"Split into {len(chunks)} chunks")
        if not chunks:
            logger.error("No chunks created. Exiting.")
            return
        # logger.info(f"First chunk: {chunks[0]}")
        
        save_to_chroma_db(chunks, logger)
        
        
        logger.info("*******************[Pipeline 1] DB population completed*******************")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        logger.debug(traceback.format_exc())


if __name__ == "__main__":
    # Create CLI.
    parser = argparse.ArgumentParser(description="Populate the database")
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    run_populate_db(reset=args.reset)