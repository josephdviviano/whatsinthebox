#!/usr/bin/env python
from pathlib import Path
from typing import Iterable
from typing import TextIO
from witb.globals import DATA_DIR, WET_URL_ROOT, WET_URL_INDEX
from witb.utils import ioutils, nlp, models
import argparse
import functools
import gzip
import io
import os
import pickle
import requests
import sys
import tempfile
import time
import urllib.request

_session = functools.lru_cache()(requests.Session)


def request_get_content(url: str, n_retry: int = 3) -> bytes:
    """Retrieve the binary content at url.
    Retry on connection errors.
    """
    t0 = time.time()
    for i in range(1, n_retry + 1):
        try:
            r = _session().get(url)
            r.raise_for_status()
            break
        except requests.exceptions.RequestException as e:
            # Sleep and try again on error, unless it's a 404.
            message = e.args[0] if isinstance(e.args[0], str) else ""
            if i == n_retry or "Client Error" in message:
                raise e

            time.sleep(10 * 2 ** i)

    dl_time = time.time() - t0
    print('Downloaded {} in {} seconds'.format(url, dl_time))

    return r.content


def open_remote_file(url: str, cache: Path = None) -> Iterable[str]:
    """Download the files at the given url to memory and opens it as a file.
    Assumes that the file is small, and fetch it when this function is called.
    """
    if cache and cache.exists():
        return open_read(cache)

    raw_bytes = request_get_content(url)
    content = io.BytesIO(raw_bytes)

    if url.endswith(".gz"):
        f: TextIO = gzip.open(content, mode="rt")  # type: ignore
    else:
        f = io.TextIOWrapper(content)

    if cache and not cache.exists():
        # The file might have been created while downloading/writing.
        tmp_cache = _tmp(cache)
        tmp_cache.write_bytes(raw_bytes)

        if not cache.exists():
            tmp_cache.replace(cache)
        else:
            tmp_cache.unlink()

    return [x.strip() for x in f.readlines()]


def get_path():
    """Full path to main.py"""
    return os.path.dirname(os.path.realpath(__file__))


def load_local_wet(filename):
    """Open a local file on disk."""
    return ioutils.get_entries(ioutils.read_wet_file(filename))


def load_remote_wet(url, idx, working_dir):
    """
    Loads the list of targets from url, and returns the data as indexed
    by idx.
    """
    remote_files = open_remote_file(url)
    remote_filename = WET_URL_ROOT + '/' + remote_files[idx]
    local_filename = os.path.join(
        working_dir.name, os.path.basename(remote_filename))

    urllib.request.urlretrieve(remote_filename, local_filename)

    return load_local_wet(local_filename)


def main(args, working_dir):

    # Clear existing output file if it exists.
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if args.overwrite and os.path.exists(args.output):
        os.remove(args.output)
    elif os.path.exists(args.output):
        sys.exit('Output {} already exists, exiting.'.format(args.output))

    # Time execution from this point forward.
    start_time = time.time()

    # Load the appropriate file, either locally or remotely.
    if args.file:
        data = load_local_wet(args.file)
    elif args.remote and args.idx:
         data = load_remote_wet(args.remote, int(args.idx), working_dir)
    else:
        sys.exit('--file or --remote & --idx input must be defined.')

    # Load the ngrams.
    wordlist_dir = os.path.join(get_path(), '../wordlists')
    wordlist_cfg = ioutils.read_yaml(os.path.join(wordlist_dir, 'config.yml'))
    ngrams = ioutils.parse_ngram_table(wordlist_dir, wordlist_cfg)

    # Generator for parsed documents.
    parsed_data = ioutils.parse_data(data)

    # Load all english documents, cleaned, into a list.
    docs = []
    for (doc, n_total_docs, n_ok_docs) in parsed_data:
        docs.append(doc)

    # Initalize the output dict.
    results = {'n_total_docs': n_total_docs, 'n_ok_docs': n_ok_docs}

    # Flag docs for matching ngrams.
    ngram_results = nlp.count_ngram_matches(docs, ngrams)

    # Hate speech / offensive text detection.
    sonar_results = nlp.run_sonar(docs)
    delimit_results = nlp.run_delimit(docs)

    # Perplexity.
    perplexity_results = nlp.run_perplexity(docs)

    print('took {} MINS to parse all valid docs'.format(
        (time.time() - start_time) / 60 ))

    # Merge all results into a single dict.
    results.update({'ngram': ngram_results,
                    'sonar': sonar_results,
                    'delimit': delimit_results,
                    'perplexity': perplexity_results})

    with open(args.output, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help='local WET file.')
    parser.add_argument('--remote', help='remote WET file list.')
    parser.add_argument('--idx', help='idx of the remote shard.')
    parser.add_argument('--output', help='output pkl filename.', required=True)
    parser.add_argument('--overwrite',
        help='overwrite output if it exists', action='store_true')
    parser.set_defaults(overwrite=False)

    args = parser.parse_args()
    working_dir = tempfile.TemporaryDirectory()

    try:
        main(args, working_dir)
    except Exception as e:
        print('run FAILED: \n{}'.format(e))

    working_dir.cleanup()

