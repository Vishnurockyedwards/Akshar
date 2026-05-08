# Hinglish Tokenizer

A complete, production-ready tokenizer designed specifically to handle Hinglish—a mix of Hindi, English, and Romanized Hindi. Built using Byte Pair Encoding (BPE) via the Hugging Face `tokenizers` library.

## Features

- **Multilingual Support**: Effectively handles English, Devanagari script (Hindi), and Romanized Hindi.
- **Emoji & Slang Preservation**: Safely tokenizes emojis and common internet slang without breaking them.
- **Robust Preprocessing**: Includes NFKC Unicode normalization and advanced lowercasing.
- **Fast and Efficient**: Powered by Hugging Face's Rust-based `tokenizers` backend.

## Project Architecture

```
hinglish-tokenizer/
├── data/                  # Sample corpora (english, hindi, roman_hindi, hinglish)
├── models/                # Saved tokenizer JSON models
├── notebooks/             # Jupyter notebooks for analysis, training, and evaluation
├── tests/                 # Unit tests (pytest)
└── tokenizer/             # Core tokenizer implementation
    ├── preprocess.py      # Text normalization and cleaning
    ├── tokenizer.py       # HinglishTokenizer class wrapper
    ├── train.py           # Training script
    └── utils.py           # Data handling utilities
```

## Installation

1. Ensure you have Python 3.8+ installed.
2. Clone this repository (or download the files).
3. Install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training the Tokenizer

You can train the tokenizer on the provided datasets using the `train.py` script:

```bash
# From the root of the project:
python -m tokenizer.train --data_dir data --output_model models/hinglish_tokenizer.json --vocab_size 130000 --min_freq 2
```

### Quick Start: Encoding and Decoding

Once the model is trained (or if you use the pre-trained one), you can use it in your Python code easily:

```python
from tokenizer.tokenizer import HinglishTokenizer

# Load the trained model
tokenizer = HinglishTokenizer.load("models/hinglish_tokenizer.json")

# Encode a custom sentence
text = "Kal meeting hai bro 😂 मैं coding सीख रहा हूँ"
encoded = tokenizer.encode(text)

print("Tokens:", encoded["tokens"])
print("Token IDs:", encoded["ids"])

# Decode back to text
decoded_text = tokenizer.decode(encoded["ids"])
print("Decoded:", decoded_text)
```

## Notebooks

We provide three Jupyter notebooks to explore the dataset and model:

1. **`dataset_analysis.ipynb`**: Understand dataset distributions and character counts.
2. **`tokenizer_training.ipynb`**: Walk through the training process step-by-step and interactively test custom sentences.
3. **`evaluation.ipynb`**: Evaluate the model to find Vocabulary Size, Unknown Token Rate, and Compression Ratio.

## Evaluation Metrics

Our evaluation tracks:
- **Vocabulary Size**: The number of unique subwords learned.
- **Unknown Token Rate (UNK %)**: How many subwords could not be mapped to the vocabulary.
- **Average Tokens per Sentence**: Average sequence length produced.
- **Compression Ratio**: Ratio of original characters to generated tokens.

## Running Tests

Run the test suite using pytest to ensure everything works correctly:

```bash
pytest tests/
```
