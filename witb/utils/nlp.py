import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import string


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


def flag_docs(docs, ngrams, threshold=3):

    flagged = {k: [] for k in ngrams.keys()}

    for doc in docs:
        bigrams = get_ngrams(doc.content, unigrams=True, bigrams=True)

        for k in ngrams.keys():
            count = sum([b in ngrams[k] for b in bigrams])

            if count > threshold:
                flagged[k].append((doc, count))

    return flagged
