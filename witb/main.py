#!/usr/bin/env python
import os
from witb.utils import ioutils as io
from witb.utils import nlp
from witb.globals import DATA_DIR
import time


def get_path():
    """Full path to main.py"""
    return os.path.dirname(os.path.realpath(__file__))


def main():

    start_time = time.time()

    # Load the ngrams.
    wordlist_dir = os.path.join(get_path(), '../wordlists')
    wordlist_cfg = io.read_yaml(os.path.join(wordlist_dir, 'config.yml'))
    ngrams = io.parse_ngram_table(wordlist_dir, wordlist_cfg)

    # Load the first WET you find.
    filenames = os.listdir(DATA_DIR)
    filename = os.path.join(DATA_DIR, filenames[0])
    data = io.get_entries(io.read_wet_file(filename))
    parsed_data = io.parse_data(data)  # Generator.

    # Load all english documents, cleaned, into a list.
    docs = []
    for doc in parsed_data:
        docs.append(doc)

    # Flag docs for matching bigrams.
    flagged = nlp.flag_docs(docs, ngrams, threshold=5)

    print('took {} MINS to parse all valid docs'.format(
        (time.time() - start_time) / 60 ))
    import IPython; IPython.embed()

if __name__ == "__main__":
    main()
