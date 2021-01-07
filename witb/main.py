#!/usr/bin/env python
import os
from witb.utils import ioutils as io
from witb.globals import DATA_DIR
import time


def main():

    # Load the first WET you find.
    filenames = os.listdir(DATA_DIR)
    filename = os.path.join(DATA_DIR, filenames[0])

    data = io.get_entries(io.read_wet_file(filename))
    parsed_data = io.parse_data(data)  # Generator.

    docs = []
    start_time = time.time()
    for doc in parsed_data:
        docs.append(doc)

    print('took {} MINS to parse all valid docs'.format(
        (time.time() - start_time) / 60 ))


if __name__ == "__main__":
    main()
