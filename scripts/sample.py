# Sample random documents. Save them to the directory samples.

import hashlib
import json
import random
from datasets import load_dataset
from pathlib import Path


def print_random_subset(n=1000):
    sample_dir = Path('samples')
    sample_dir.mkdir(parents=True, exist_ok=True)

    start = random.randint(0, 26_000_000)
    print(f'Taking {n} documents starting randomly at {start}')

    dataset = load_dataset('mc4', 'fi', split='train', streaming=True).skip(start).take(n)
    for datapoint in dataset:
        save_sample_document(datapoint, sample_dir)


def save_sample_document(datapoint, sample_path):
    docstr = json.dumps(datapoint, ensure_ascii=False)
    sha = hashlib.sha256()
    sha.update(docstr.encode('UTF-8'))
    docid = sha.hexdigest()
    try:
        with open(sample_path / docid, 'w') as f:
            f.write(docstr)
    except IOError:
        pass


if __name__ == '__main__':
    print_random_subset()
