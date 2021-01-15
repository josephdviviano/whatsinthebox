#from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from hatesonar import Sonar
from multiprocessing import Pool
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import multiprocessing
import numpy as np
import pandas as pd
import torch


class SonarRunner():
    def __init__(self, threshold=20):
        self._model = Sonar()
        self.labels = {
            'neither': 0,
            'offensive_language': 1,
            'hate_speech': 2}
        self.threshold = threshold

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

        scores /= n  # Take the mean.

        return np.concatenate([labels, scores])


class DeLimitRunner():
    def __init__(self, threshold=20):
        self._model = AutoModelForSequenceClassification.from_pretrained(
            "Hate-speech-CNERG/dehatebert-mono-english")
        self.labels = {
            'hate_speech': 0,
            'normal': 1}
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Hate-speech-CNERG/dehatebert-mono-english")
        self.threshold = threshold

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device('cpu')

        self.model.to(self.device)

    def query(self, doc):
        """Runs all sentences across cores."""
        # TODO - not sure how to do this
        labels = np.zeros(len(self.labels))  # Per sentence counts of labels.
        scores = np.zeros(len(self.labels))  # Mean score per label.

        # Preprocess sentences (filter, add delimiters, tokenize).
        sentences = [s for s in doc.sentences if len(s) > self.threshold]
        sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
        sentences = [self.tokenizer.tokenize(s, padding='max_length', truncation=True) for s in sentences]
        sentences = [self.tokenizer.convert_tokens_to_ids(s) for s in sentences]
        n = len(sentences)

        # Create attention masks (check if this works correctly)
        attention_masks = [float(i>0) for i in input_ids]

        if n == 0:
            return np.concatenate([labels, scores])

        #New stuff
        #I'm assigning a 'non-hate' label to each sentence as a proxy

        # TODO: We are most likely going to remove this stuff unless we really
        #       need the dataloader to go fast (unclear).
        #prediction_data = TensorDataset(prediction_inputs, prediction_masks)
        #do we need a sampler? not sure
        #prediction_sampler = SequentialSampler(prediction_data)
        #prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=5, num_workers=1)

        for sentence, mask in zip(sentences, attention_masks):
            sentence = torch.tensor(sentence).to(self.device)
            mask = torch.tensor(mask).to(self.device)

            with torch.no_grad():
                # TODO: might need to unsqueeze the batch dim.
                logits = self.model(
                    sentence, token_type_ids=None, attention_mask=mask)[0]

            logits = logits.detach().cpu().numpy()

            # TODO: we need to append these results to lists.
            #       One output per DOCUMENT, so combine results over sentences.
            pred_labels += list(np.argmax(logits, axis=1).flatten())

            #results = logits
            #label = pred_labels
            
            for r in logits:
                scores[self.labels[r['class_name']]] += r['confidence']

        scores /= n  # Take the mean.


            # TODO Gets the numeric score for each class per sentence.
            #for r in result['classes']:
            #   scores[self.labels[r['class_name']]] += r['confidence']

        #scores /= n

        return np.concatenate([labels, results])



# XPLAIN
#    "0": "hate speech",
#    "1": "normal",
#    "2": "offensive"
