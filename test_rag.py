import os
import yaml
from logger import setup_logger
from pipeline.stage_02_query_data import query_rag
from langchain_ollama import OllamaLLM


# load parameters from "params.yaml"
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

LOG_PATH = params["LOG_PATH"]
GENERATION_MODEL = params["GENERATION_MODEL"]


# setup logging
LOG_DIR = os.path.join(os.getcwd(), LOG_PATH)
os.makedirs(LOG_DIR, exist_ok=True)  # Create the logs directory if it doesn't exist
LOG_FILE = os.path.join(LOG_DIR, "test_rag.log")


EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""


def test_monopoly_rules():
    assert query_and_validate(
        question="How much total money does a player start with in Monopoly? (Answer with the number only)",
        expected_response="$1500",
    )
    assert not query_and_validate(
        question="How much total money does a player start with in Monopoly? (Answer with the number only)",
        expected_response="$9999",
    )


def test_ticket_to_ride_rules():
    assert query_and_validate(
        question="How many points does the longest continuous train get in Ticket to Ride? (Answer with the number only)",
        expected_response="10",
    )


def query_and_validate(question: str, expected_response: str):
    logger = setup_logger("test_logger", LOG_FILE)
    logger.info(" ")
    
    logger.info("*******************Running test*******************")
    response_text = query_rag(question, logger)
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, 
        actual_response=response_text
    )

    model = OllamaLLM(model=GENERATION_MODEL)
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    print(prompt)

    if "true" in evaluation_results_str_cleaned:
        # Print response in Green if it is correct.
        print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        logger.info(f"Response: {evaluation_results_str_cleaned}")
        logger.info("*******************Test passed*******************")
        return True
    elif "false" in evaluation_results_str_cleaned:
        # Print response in Red if it is incorrect.
        print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
        logger.info(f"Response: {evaluation_results_str_cleaned}")
        logger.info("*******************Test failed*******************")
        return False
    else:
        raise ValueError(
            f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
        )

# if __name__ == "__main__":
#     print("Running test_monopoly_rules()")
#     test_monopoly_rules()

#     print("Running test_ticket_to_ride_rules()")
#     test_ticket_to_ride_rules()