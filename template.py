import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')


project_name = "rag_example_2"

list_of_files = [
    "data/.gitkeep",
    "chromadb/.gitkeep",
    "models/.gitkeep",
    
    "notebooks/trials.ipynb",
    "templates/index.html",
    
    "pipeline/__init__.py",
    "pipeline/01_populate_db.py",
    "pipeline/02_query_data.py",
    "pipeline/03_create_response.py",
    "pipeline/04_evaluate.py",
    
    "main.py",
    "app.py",
    "utils.py",
    
    "config.yaml",
    "params.yaml",
    "DVC.yaml",
    ".env.local",
    "requirements.txt",
]


for filepath in list_of_files:
    filepath = Path(filepath) #to solve the windows path issue
    filedir, filename = os.path.split(filepath) # to handle the project_name folder


    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")


    else:
        logging.info(f"{filename} is already exists")