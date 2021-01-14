#!/usr/bin/env python

from typing import List
from witb.globals import WET_URL_ROOT
from bs4 import BeautifulSoup
import urllib.request
import re

def cc_wet_paths_url(dump_id: str) -> str:
    return "/".join([WET_URL_ROOT,
                     "crawl-data",
                     "CC-MAIN-" + dump_id,
                     "wet.paths.gz"])


def list_dumps() -> List[str]:
    home_page = BeautifulSoup(
        urllib.request.urlopen("http://index.commoncrawl.org/"), features="html.parser"
    )
    dumps = [a.get("href").strip("/") for a in home_page.findAll("a")]
    dumps = [a[8:] for a in dumps if re.match(r"^CC-MAIN-\d\d\d\d-\d\d$", a)]

    return sorted(dumps)


def main():
    dumps = list_dumps()
    for dump in dumps:
        print(cc_wet_paths_url(dump))


if __name__ == "__main__":
    main()

