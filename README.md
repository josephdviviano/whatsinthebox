whats in the box?
-----------------

Tools associated with
What's in the Box? A Preliminary Analysis of Undesirable Content in the Common Crawl Corpus
Alexandra Sasha Luccioni, Joseph D. Viviano
ACL-IJCNLP 2021
https://arxiv.org/abs/2105.02732

This performs a simple analysis of common NLP corpora used for training language models.


**intstallation**

`conda env create -f environment.yml`

NOTE: This project ran into environment issues since we were running many
different published models against the corpus, and `environment.yml` does not
capture all of the dependencies as we generated different environments on different
systems to produce the results. This is unfortunate, and if anyone wants to use
the model wrappers from this tool and have issues, please file an issue here
and I can try to help debug.

**quickstart**

To see all remote common crawl data available:

```
witb/list_cc.py
```

The `idx` parameter can be used to process a single `wet` file from the common crawl.
Therefore, it is very easy to processes many files in parallel across multiple machines
on a compute cluster (simply submitting jobs looping over `idx` should be sufficient).

See below for example usage for a single `idx`. See `data/example.pkl` for an
example output file.

```
witb/main.py \
  --remote=https://commoncrawl.s3.amazonaws.com/crawl-data/CC-MAIN-2020-50/wet.paths.gz \
  --idx=0 \
  --output=outputs/test.pkl \
  --overwrite
```

