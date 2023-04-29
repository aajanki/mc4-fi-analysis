import hashlib
import json
import random
import re
import unicodedata
import langid
import spacy
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse
from datasets import load_dataset
from ftlangdetect import detect


def main():
    output_path = Path('results')
    output_path.mkdir(parents=True, exist_ok=True)
    sample_path = Path('samples')
    sample_path.mkdir(parents=True, exist_ok=True)

    doc_count = 0
    domain_hist = Counter()
    date_hist = Counter()
    token_count_hist: defaultdict[int, int] = defaultdict(int)
    fi_detection_hist = {
        'fi-fi': 0,
        'fi-other': 0,
        'other-fi': 0,
        'other-other': 0
    }

    nlp = create_nlp()
    dataset = load_dataset('mc4', 'fi', split='train', streaming=True)
    for datapoint in dataset:
        if random.random() < 1e-5:
            save_sample_document(datapoint, sample_path)

        doc_count += 1
        text = unicodedata.normalize('NFC', datapoint['text'].strip())
        text = re.sub(r'\s+', ' ', text)

        # number of tokens
        doc = nlp(text)
        num_tokens = len(doc)
        token_count_hist[num_tokens] += 1

        # language detection
        if langid.classify(text)[0] == 'fi':
            langid_key = 'fi'
        else:
            langid_key = 'other'

        if detect(text.replace('\n', ' '))['lang'] == 'fi':
            fasttext_key = 'fi'
        else:
            fasttext_key = 'other'

        fi_detection_key = f'{langid_key}-{fasttext_key}'
        fi_detection_hist[fi_detection_key] += 1

        # Domain
        url = datapoint['url']
        netloc = domain(url)
        if netloc:
            domain_hist.update([netloc])

        # Timestemp
        timestamp = datapoint['timestamp']
        datestr = timestamp.split('T')[0]
        if datestr:
            date_hist.update([datestr])

        if doc_count % 100000 == 0:
            save_statistics(domain_hist, date_hist, token_count_hist, fi_detection_hist, doc_count, output_path)

        if doc_count % 100000 == 0:
            del nlp
            nlp = create_nlp()

    save_statistics(domain_hist, date_hist, token_count_hist, fi_detection_hist, doc_count, output_path)


def create_nlp():
    return spacy.load('fi_core_news_sm', exclude=[
        'tok2vec', 'tagger', 'parser', 'lemmatizer', 'morphologizer', 'ner'
    ])


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


def save_statistics(domain_hist, date_hist, token_count_hist, fi_detection_hist, doc_count, output_path):
    print(f'Saving statistics after {doc_count} documents to {output_path}')

    with open(output_path / 'domains.tsv', 'w') as f:
        for netloc, freq in domain_hist.most_common():
            f.write(f'{netloc}\t{freq}\n')

    with open(output_path / 'date.tsv', 'w') as f:
        for datestr, freq in date_hist.most_common():
            f.write(f'{datestr}\t{freq}\n')

    with open(output_path / 'tokens.tsv', 'w') as f:
        for num_tokens, freq in token_count_hist.items():
            f.write(f'{num_tokens}\t{freq}\n')

    with open(output_path / 'language_detection.tsv', 'w') as f:
        for lang, freq in fi_detection_hist.items():
            f.write(f'{lang}\t{freq}\n')


def domain(url: str) -> str:
    parsed_url = urlparse(url)
    netloc = parsed_url.netloc.lower()
    # Remove port
    netloc = netloc.split(':', 1)[0]
    return netloc


if __name__ == '__main__':
    main()
