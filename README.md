whats in the box?
-----------------

A simple analysis of common NLP corpora used for training language models.

TODO:
+ Filter language.
+ Deduplicate.
+ Analyze a sample of raw WET files.
+ Analyze a sample of raw OSCAR stuffs.

**intstallation**

`pip install -e .`

**shows all remote common crawl data available**

`witb/list_cc.py`

**example usage**

`witb/main.py --remote=https://commoncrawl.s3.amazonaws.com/crawl-data/CC-MAIN-2020-50/wet.paths.gz --idx=0 --output=outputs/test.pkl --overwrite`

