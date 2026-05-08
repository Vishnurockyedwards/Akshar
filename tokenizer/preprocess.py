import unicodedata
import re
import emoji

def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode text using NFKC.
    This ensures that visually similar characters are unified.
    """
    return unicodedata.normalize('NFKC', text)

def lowercase_text(text: str) -> str:
    """
    Convert text to lowercase.
    """
    return text.lower()

def preserve_emojis(text: str) -> str:
    """
    Ensure emojis are separated by spaces so the tokenizer treats them as individual tokens.
    """
    return emoji.replace_emoji(text, replace=lambda chars, data_dict: f' {chars} ')

def clean_punctuation(text: str) -> str:
    """
    Normalize repeating punctuation or clean unnecessary symbols if needed.
    (Currently a placeholder for advanced punctuation logic).
    """
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def preprocess_text(text: str) -> str:
    """
    Run all preprocessing steps on the given text.
    """
    text = normalize_unicode(text)
    text = preserve_emojis(text)
    text = lowercase_text(text)
    text = clean_punctuation(text)
    return text
