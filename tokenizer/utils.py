import os
from typing import List
from .preprocess import preprocess_text

def load_data(file_path: str) -> List[str]:
    """
    Load data from a text file, line by line.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def load_and_preprocess_corpus(data_dir: str) -> List[str]:
    """
    Load all text files from a directory, merge them, and preprocess the text.
    """
    corpus = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_dir, filename)
            lines = load_data(file_path)
            # Preprocess each line
            corpus.extend([preprocess_text(line) for line in lines])
            
    return corpus

def save_corpus(corpus: List[str], output_path: str):
    """
    Save the merged and preprocessed corpus to a file for tokenizer training.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in corpus:
            f.write(line + '\n')
