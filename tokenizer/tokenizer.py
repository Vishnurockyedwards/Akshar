import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from .preprocess import preprocess_text
from typing import List, Union

class HinglishTokenizer:
    """
    A tokenizer for Hinglish (Hindi + English + Roman Hindi), built using
    the Hugging Face tokenizers library with Byte Pair Encoding (BPE).
    """
    def __init__(self, tokenizer: Tokenizer = None):
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
            self.tokenizer.decoder = ByteLevelDecoder()
            
        self.special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    def train_from_file(self, file_path: str, vocab_size: int = 30000, min_frequency: int = 2):
        """
        Train the tokenizer from a text file.
        """
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=self.special_tokens
        )
        self.tokenizer.train([file_path], trainer)

    def encode(self, text: str) -> dict:
        """
        Encode a string into token IDs.
        """
        preprocessed = preprocess_text(text)
        encoding = self.tokenizer.encode(preprocessed)
        return {
            "tokens": encoding.tokens,
            "ids": encoding.ids,
            "attention_mask": encoding.attention_mask
        }

    def decode(self, token_ids: Union[List[int], int]) -> str:
        """
        Decode token IDs back into a string.
        """
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return self.tokenizer.decode(token_ids)

    def save(self, path: str):
        """
        Save the tokenizer model to a file.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.tokenizer.save(path)

    @classmethod
    def load(cls, path: str) -> 'HinglishTokenizer':
        """
        Load a trained tokenizer from a file.
        """
        hf_tokenizer = Tokenizer.from_file(path)
        return cls(tokenizer=hf_tokenizer)

    def get_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()
