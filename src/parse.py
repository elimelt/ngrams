import os
import re
from pathlib import Path
from collections import defaultdict, Counter
import csv
from typing import Dict, List, Tuple

DATA_ROOT = "data"
N = 5  # n-gram size (includes prediction character)

def normalize_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return text.strip()

def update_ngram_counts(text: str, counts: Dict[str, Counter]) -> None:
    for i in range(len(text) - N + 1):
        ngram = text[i:i+N]
        context, char = ngram[:-1], ngram[-1]
        counts[context][char] += 1

def parse_file(path: str, chunk_size: int = 1024*1024) -> Dict[str, Counter]:
    counts = defaultdict(Counter)
    with open(path, 'r', encoding='utf-8') as f:
        while chunk := f.read(chunk_size):
            text = normalize_text(chunk)
            update_ngram_counts(text, counts)
    return counts

def write_ngram_csv(counts: Dict[str, Counter], output_path: str) -> None:
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for context, char_counts in counts.items():
            top_3 = char_counts.most_common(3)
            chars = ''.join(char for char, _ in top_3)
            writer.writerow([context, chars])

if __name__ == "__main__":
    total_counts = defaultdict(Counter)
    for path in Path(DATA_ROOT).rglob('*'):
        if path.is_file():
            print(f"Processing {path}")
            file_counts = parse_file(str(path))
            for context, counts in file_counts.items():
                total_counts[context].update(counts)

    write_ngram_csv(total_counts, 'work/model.csv')