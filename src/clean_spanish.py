from collections import Counter, defaultdict
import os
from pathlib import Path
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def process_file(path: str, out_dir: str):
    print(path)
    contents = None
    with open(path, encoding="latin-1") as f:
        contents = f.readlines()
    
    _, tail = os.path.split(path)
    out_path = os.path.join(out_dir, tail)
    with open(out_path, "w", encoding="utf-8") as o:
        for line in contents:
            if not line.strip() or line.startswith("<doc") or line.startswith("</doc>") or line.startswith("ENDOFARTICLE"):
                continue
            
            line = line.replace(" .", ".")
            o.write(line.lstrip())


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--in_dir', help='path to root of the 120 Million Word Spanish Corpus for processing')
    parser.add_argument('--out_dir', help='path to save clean data to', default="data/SpanishCorpus")
    args = parser.parse_args()
    
    for path in Path(args.in_dir).rglob("*"):
        if path.is_file():
            process_file(path, args.out_dir)
            