import numpy as np
from hatesonar import Sonar
import multiprocessing
from multiprocessing import Pool


class SonarRunner():
    def __init__(self, threshold=20):
        self._model = Sonar()
        self.labels = {
            'neither': 0,
            'offensive_language': 1,
            'hate_speech': 2}
        self.threshold = threshold

    def proc_sentence(self, sentence):
        """Results for a single sentence."""
        labels = np.zeros(len(self.labels))
        scores = np.zeros(len(self.labels))


        return (labels, scores)

    def query(self, doc):
        """Runs all sentences across cores."""
        labels = np.zeros(len(self.labels))  # Per sentence counts of labels.
        scores = np.zeros(len(self.labels))  # Mean score per label.

        # Remove short sentences.
        sentences = [s for s in doc.sentences if len(s) > self.threshold]
        n = len(sentences)

        if n == 0:
            return np.concatenate([labels, scores])

        for sentence in sentences:
            result = self._model.ping(text=sentence)
            labels[self.labels[result['top_class']]] += 1  # Top class count.

            # Gets the numeric score for each class per sentence.
            for r in result['classes']:
                scores[self.labels[r['class_name']]] += r['confidence']

        scores /= n

        return np.concatenate([labels, scores])


#DE-LIMIT
#0 : hate speech
#1 : normal


# XPLAIN
#    "0": "hate speech",
#    "1": "normal",
#    "2": "offensive"
