import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from witb.utils import models
import re
import string
import numpy as np
import multiprocessing
from multiprocessing import Pool


class Cleaner():
    def __init__(self):
        self._stopwords = set(stopwords.words('english'))

    def clean(self, text):
        """Return cleaned text as a list."""
        output = []
        for word in  word_tokenize(text):
            if word in self._stopwords:
                continue  # Skip stopwords.
            else:
                # Remove numbers, punctuation, and lowercase all text.
                stripped_word = word.strip(
                    string.digits + string.punctuation).lower()
                if stripped_word:
                    output.append(stripped_word)  # Skip empty words.

        return output


def get_ngrams(text, unigrams=False, bigrams=True, trigrams=False):
    """Returns a list of unigrams, bigrams, and trigrams from the input text."""
    ngrams = []

    if unigrams:
        ngrams.extend([tuple( (w,) ) for w in text])
    if bigrams:
        ngrams.extend([b for b in nltk.bigrams(text)])
    if trigrams:
        ngrams.extend([b for b in nltk.trigrams(text)])

    return ngrams


def count_ngram_matches(docs, ngrams):

    flagged = {k: [] for k in ngrams.keys()}

    for doc in docs:
        bigrams = get_ngrams(doc.content, unigrams=True, bigrams=True)
        for k in ngrams.keys():
            count = sum([b in ngrams[k] for b in bigrams])
            if k == "hate-speech" and count >3 :
                for b in ngrams[k]:
                    if b in bigrams:
                        print('NGRAMS ', b)
            flagged[k].append(count)

    return flagged


def run_perplexity(docs):
    perplexity = models.PerplexRunner()

    results = []
    for doc in docs:
        results.append(perplexity.query(doc))

    return np.vstack(results)


def run_sonar(docs):

    n_workers = multiprocessing.cpu_count()
    p = Pool(n_workers)

    sonar = models.SonarRunner()
    results = p.map(sonar.query, docs)

    return np.vstack(results)


def run_delimit(docs):

    #n_workers = multiprocessing.cpu_count()
    #p = Pool(n_workers)

    delimit = models.DeLimitRunner()
    results = []
    for doc in docs:
        results.append(delimit.query(doc))

    #results = p.map(delimit.query, docs)

    return np.vstack(results)

'''

def run_sonar(docs):

    n_workers = multiprocessing.cpu_count()
    p = Pool(n_workers)

    sonar = models.SonarRunner()
    results = p.map(sonar.query, docs)

    return np.vstack(results)

def run_sonar(docs):

    n_workers = multiprocessing.cpu_count()
    p = Pool(n_workers)

    sonar = models.SonarRunner()
    results = p.map(sonar.query, docs)

    return np.vstack(results)

'''
