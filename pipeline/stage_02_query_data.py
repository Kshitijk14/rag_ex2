import os
import yaml
import logging
import traceback
import argparse
from logger import setup_logger
from langchain_chroma import Chroma
from get_embedding_func import embedding_function
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM


# load parameters from "params.yaml"
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

LOG_PATH = params["LOG_PATH"]
GAME_RULES_DATA_PATH = params["GAME_RULES_DATA_PATH"]
CHROMA_DB_PATH = params["CHROMA_DB_PATH"]
CHROMA_DB_FILE = params["CHROMA_DB_FILE"]
GENERATION_MODEL = params["GENERATION_MODEL"]

# setup logging
LOG_DIR = os.path.join(os.getcwd(), LOG_PATH)
os.makedirs(LOG_DIR, exist_ok=True)  # Create the logs directory if it doesn't exist
LOG_FILE = os.path.join(LOG_DIR, "02_query_data.log")


PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def query_rag(query_text: str, logger):
    try:
        logger.info("*******************[Pipeline 2] Querying Chroma DB*******************")
        
        # load the existing db (prep the db)
        db = Chroma(
            embedding_function=embedding_function(),
            persist_directory=CHROMA_DB_PATH,
        )
        logger.info(f"[Stage 1] Loading existing DB from path: {CHROMA_DB_PATH}")
        
        # query the db (search the db)
        logger.info(f"[Stage 2] Searching the db with text: {query_text}")
        results = db.similarity_search_with_score(query_text, k=5)
        
        logger.info("*******************[Pipeline 2] Querying Chroma DB completed successfully*******************")
        
        logger.info("*******************[Pipeline 3] Generating response from LLM (based on context)*******************")
        
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        logger.info(f"[Stage 3] Prompting LLM with the context")
        # logger.info(f"[Stage 3] Prompting LLM with text: {prompt}")
        
        model = OllamaLLM(model=GENERATION_MODEL)
        response_text = model.invoke(prompt)
        logger.info(f"[Stage 4] LLM's response: {response_text}")
        
        sources = [doc.metadata.get("chunk_id", None) for doc, _score in results]
        formatted_response = (f"Response: {response_text}\nSources: {sources}")
        logger.info(f"[Stage 5] Answer: {formatted_response}")
        
        logger.info("*******************[Pipeline 3] Generating response from LLM completed successfully*******************")
        
        return response_text
    except Exception as e:
        logger.error(f"Error querying Chroma DB: {e}")
        logger.debug(traceback.format_exc())
        return []


def run_query_rag(query_text):
    logger = setup_logger("create_db_logger", LOG_FILE)
    logger.info(" ")
    
    return query_rag(query_text, logger)

if __name__ == "__main__":
    # Create CLI.
    parser = argparse.ArgumentParser(description="Query the Chroma DB.")
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    run_query_rag(args.query_text)