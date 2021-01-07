#!/usr/bin/env python
import os
import sys
import gzip
from witb.utils import io
from witb.globals import DATA_DIR, HEADER
import time


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


def main():

    # Load the first WET you find.
    filenames = os.listdir(DATA_DIR)
    filename = os.path.join(DATA_DIR, filenames[0])

    data = get_entries(read_wet_file(filename))
    parsed_data = io.parse_data(data)

    docs = []
    start_time = time.time()
    for doc in parsed_data:
        docs.append(doc)

    print('took {} MINS to parse all valid docs'.format(
        (time.time() - start_time) / 60 ))


if __name__ == "__main__":
    main()
