#!/usr/bin/env python
from collections import Counter, defaultdict
import ctypes
import os
from pathlib import Path
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tqdm import tqdm
import mmap

import csv
import re

N = 5
DATA_ROOT = 'data'
MODEL_PATH = 'work/model.csv'

class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """

    def __init__(self):
        self.lookups = None
        self.counts = None

    @staticmethod
    def pad_prediction(pred):
        """pad prediction to exactly 3 characters using e, s, space."""
        if len(pred) >= 3:
            return pred[:3]
        return pred + "es "[:(3 - len(pred))]

    @staticmethod
    def normalize_text(text: str) -> str:
        """lowercase and removing extra spaces."""
        return re.sub(r"\s+", " ", text).lower().strip()

    @classmethod
    def load_training_data(cls):
        data = []
        for path in tqdm(Path(DATA_ROOT).rglob("*")):
            if path.is_file() and not str(path).endswith(".DS_Store"):
                data.append(MyModel.normalize_text(path.read_text(encoding="utf-8")))
        return data

    @classmethod
    def load_test_data(cls, fname):
        with open(fname, "r") as f:
            # mmap can be slightly faster than readlines
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                return mm.read().decode("utf-8").splitlines(True)

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            f.writelines(preds)

    def train_ngrams(self, train_data, n):
        for text in tqdm(train_data):
            for i in range(len(text) - n + 1):
                ngram = text[i : i + n]
                context, char = ngram[:-1], ngram[-1]
                self.counts[context][char] += 1

    def run_train(self, data, work_dir):
        self.counts = defaultdict(Counter)
        for n in range(1, N + 1):
            print("N:", n)
            self.train_ngrams(data, n)
        self.lookups = {}
        self.save(work_dir)
        self.load(work_dir)

    def run_pred(self, data):
        preds = [None] * len(data)
        lookups = self.lookups
        pad_str = "es "

        for idx, inp in enumerate(data):
            inp = inp[:-1].lower()
            n = N
            pred = ""

            while n > 0 and len(pred) < 3:
                # pad prefix to n-1 characters long
                prefix = (" " * (n - 1 - len(inp)) + inp[-(n - 1):]) if len(inp) < (n - 1) else inp[-(n - 1):]
                if prefix in lookups:
                    for c in lookups[prefix]:
                        if c not in pred:
                            pred += c
                            if len(pred) >= 3:
                                break
                    if len(pred) >= 3:
                        break
                n -= 1

            # pad prediction
            preds[idx] = (pred + "es ")[:3] + "\n"

        return preds

    def save(self, work_dir):
        if not self.counts:
            raise ValueError('Tried to save model before training')

        with open(os.path.join(work_dir, "model.csv"), 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(
                f, quoting=csv.QUOTE_NONNUMERIC, doublequote=True, escapechar=None
            )
            for context, char_counts in self.counts.items():
                chars = "".join(char for char, _ in char_counts.most_common(3))
                if "\u0000" not in context and "\u0000" not in chars:
                    writer.writerow([context, chars])

    @classmethod
    def load(cls, work_dir):
        # TODO: remove check to save more time
        model_path = os.path.join(work_dir, "model.csv")
        if not os.path.isfile(model_path):
            raise ValueError(f'No model found in {work_dir}')

        pad_prediction = cls.pad_prediction
        # pre-size lookups dict
        ctypes.pythonapi._PyDict_NewPresized.restype = ctypes.py_object
        lookups = ctypes.pythonapi._PyDict_NewPresized(120_000)

        # buffer file reading to 1MB
        # TODO: set it to be slightly larger than exact model.csv size of submission
        with open(model_path, encoding="utf-8", buffering=1024*1024) as preds_csv:
            preds_reader = csv.reader(
                preds_csv,
                delimiter=',',
                quoting=csv.QUOTE_NONNUMERIC,
                doublequote=True,
                escapechar=None
            )
            for bigram, preds in preds_reader:
                lookups[bigram] = pad_prediction(preds)

        model = cls()
        model.lookups = lookups
        return model


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
