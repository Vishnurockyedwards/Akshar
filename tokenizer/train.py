import os
import argparse
from tokenizer.utils import load_and_preprocess_corpus, save_corpus
from tokenizer.tokenizer import HinglishTokenizer

def main():
    parser = argparse.ArgumentParser(description="Train Hinglish Tokenizer")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing text files")
    parser.add_argument("--output_model", type=str, default="models/hinglish_tokenizer.json", help="Path to save the trained model")
    parser.add_argument("--vocab_size", type=int, default=30000, help="Vocabulary size")
    parser.add_argument("--min_freq", type=int, default=2, help="Minimum frequency for a pair")
    
    args = parser.parse_args()

    print(f"Loading and preprocessing data from {args.data_dir}...")
    corpus = load_and_preprocess_corpus(args.data_dir)
    
    # Save the merged corpus temporarily
    temp_corpus_path = os.path.join(args.data_dir, "merged_corpus_temp.txt")
    save_corpus(corpus, temp_corpus_path)
    
    print(f"Training tokenizer with vocab_size={args.vocab_size}, min_freq={args.min_freq}...")
    tokenizer = HinglishTokenizer()
    tokenizer.train_from_file(
        file_path=temp_corpus_path,
        vocab_size=args.vocab_size,
        min_frequency=args.min_freq
    )
    
    print(f"Saving tokenizer to {args.output_model}...")
    tokenizer.save(args.output_model)
    
    # Cleanup temp file
    if os.path.exists(temp_corpus_path):
        os.remove(temp_corpus_path)
        
    print("Training complete!")

if __name__ == "__main__":
    main()
