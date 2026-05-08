import os
import pytest
from tokenizer.tokenizer import HinglishTokenizer
from tokenizer.preprocess import preprocess_text

@pytest.fixture(scope="module")
def tokenizer():
    # Setup: Train a quick tokenizer for testing
    data_dir = "data"
    temp_corpus_path = "test_corpus.txt"
    with open(temp_corpus_path, "w", encoding="utf-8") as f:
        f.write("Kal meeting hai bro 😂\n")
        f.write("मैं coding सीख रहा हूँ\n")
        f.write("Aaj weather bahut awesome hai 😂\n")
        f.write("Bro assignment submit kar diya kya?\n")

    tok = HinglishTokenizer()
    tok.train_from_file(temp_corpus_path, vocab_size=1000, min_frequency=1)
    
    yield tok
    
    # Teardown
    if os.path.exists(temp_corpus_path):
        os.remove(temp_corpus_path)

def test_preprocessing():
    # Test emoji preservation and lowercasing
    text = "Hello World 😂"
    processed = preprocess_text(text)
    assert "😂" in processed
    assert "hello world" in processed
    assert " 😂 " in processed or "😂 " in processed or " 😂" in processed # Depending on space replacement

def test_encode_decode(tokenizer):
    text = "kal meeting hai"
    encoded = tokenizer.encode(text)
    
    assert "tokens" in encoded
    assert "ids" in encoded
    
    decoded = tokenizer.decode(encoded["ids"])
    assert decoded.strip() == text.strip()

def test_hindi_encoding(tokenizer):
    text = "मैं coding सीख"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded["ids"])
    # Note: BPE might add spaces depending on ByteLevel prefix, so we check inclusion
    assert "coding" in decoded
    assert "मैं" in decoded

def test_emoji_encoding(tokenizer):
    text = "awesome hai 😂"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded["ids"])
    assert "😂" in decoded

def test_save_load(tokenizer):
    model_path = "models/test_tokenizer.json"
    tokenizer.save(model_path)
    
    assert os.path.exists(model_path)
    
    loaded_tok = HinglishTokenizer.load(model_path)
    assert loaded_tok.get_vocab_size() > 0
    
    if os.path.exists(model_path):
        os.remove(model_path)
