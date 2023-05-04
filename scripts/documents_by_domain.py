# Download and print mc4 documents from the given domain.
#
# For example:
# python documents_by_domain.py www.hs.fi

import json
import sys
from urllib.parse import urlparse
from datasets import load_dataset


def main():
    if len(sys.argv) < 2:
        sys.exit(1)

    print_by_domain(sys.argv[1])


def print_by_domain(domain_to_find):
    dataset = load_dataset('mc4', 'fi', split='train', streaming=True)
    for datapoint in dataset:
        url = datapoint['url']
        if domain(url) == domain_to_find:
            print(json.dumps(datapoint, ensure_ascii=False))


def domain(url: str) -> str:
    parsed_url = urlparse(url)
    netloc = parsed_url.netloc.lower()
    # Remove port
    netloc = netloc.split(':', 1)[0]
    return netloc


if __name__ == '__main__':
    main()
