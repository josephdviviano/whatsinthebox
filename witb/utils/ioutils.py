# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from urllib.parse import urlparse
from witb.globals import HEADER
from witb.utils import nlp
import csv
import gzip
import logging
import yaml
import os

logger = logging.getLogger(__name__)


class Document():
    def __init__(self, entry):

        self.sentences = []
        self._header = {
            'WARC-Type': None,
            'WARC-Target-URI': None,
            'WARC-Date': None,
            'WARC-Filename': None,
            'WARC-Record-ID': None,
            'WARC-Refers-To': None,
            'WARC-Identified-Content-Language': None,
            'Content-Type': None,
            'Content-Length': None,
            'Software-Info': None,
            'Extracted-Date': None,
            'robots': None,
            'isPartOf': None,
            'operator': None,
            'description': None,
            'publisher': None}

        header = True
        for line in entry:
            # The header and body are seperated by a single line, and the
            # content is followed by two empty lines.
            if line == '':
                header = False
            # Fill the appropriate header value.
            elif header:
                line = line.split(':')
                prop, val = line[0], ':'.join(line[1:])
                self._header[prop] = val.strip()
            # Populate the content.
            else:
                self.sentences.append(line)

        # Merge the content into a single string.
        self.content = '\n'.join(self.sentences)

        # Useful metadata.
        self.url = urlparse(self._header['WARC-Target-URI']).netloc
        self.language = self._header['WARC-Identified-Content-Language']

    def __repr__(self):
        if 0 < len(self.content) < 100:
            return ' '.join(self.content)
        elif len(self.content) > 100:
            return ' '.join(self.content[:100]) + '...'
        else:
            return 'N/A'


def parse_data(entries, min_len=1, languages=['eng']):
    """Yield each entry as a structured and pre-processed object."""

    n_doc, n_ok = len(entries), 0
    cleaner = nlp.Cleaner()  # Prepares the text.

    for entry in entries:
        doc = Document(entry)
        if len(doc.content) > min_len and doc.language in languages:
            doc.content = cleaner.clean(doc.content)
            n_ok += 1
            yield (doc, n_doc, n_ok)

    if n_doc > 0:
        logger.info(
            f"Kept {n_ok:_d} documents over {n_doc:_d} ({n_ok / n_doc:.1%}).")
    else:
        logger.info("Found no documents")


def read_wet_file(wet_file, max_lines=-1):
    """
    Args:
        wet_file (str): path to input WET file (gz format).
        max_lines (int): maximum number of lines to read.
    Returns: WET file in the form of a list.
    """
    output = []
    with gzip.open(wet_file, mode='rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            output.append(line.strip())
            if i > max_lines and max_lines > 0:
                break

    return output


def get_entries(data):
    """Split the text into lists of lists."""
    entries, entry = [], []

    for i, line in enumerate(data):
        if line.startswith(HEADER):
            if len(entry) > 0:
                entries.append(entry)
            entry = []
        else:
            entry.append(line)

    return entries


def read_yaml(yaml_file):
    with open(yaml_file) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def parse_ngram_table(base_path, cfg):
    """Parse all ngrams defined in the configuration into a hash table.
    Args:
        base_path (str): path of the wordlist configuration file.
        cfg (dict): parsed wordlist configuration file.
    Returns: A hash table of all parsed ngrams defined in the config, organized
        by top-level keys.
    """
    ngram_table = {}
    cfg = cfg['ngrams']

    for k, v in cfg.items():
        ngram_table[k] = {}  # Table is organized by category k.

        # Each filename is expected to be a list of ngrams.
        for filename in v:
            ngrams = parse_ngram_file(os.path.join(base_path, filename))
            ngram_table[k].update(ngrams)

    return ngram_table


def parse_ngram_file(csv_file):
    """Parse a single ngram wordlist into a hashtable."""
    ngrams = {}
    with open(csv_file, newline='') as f:
        for row in csv.reader(f):
            ngram = tuple(row[0].split())
            ngrams[ngram] = True

    return ngrams
