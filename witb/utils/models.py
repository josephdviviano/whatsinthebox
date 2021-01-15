#from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from hatesonar import Sonar
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')



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
        self._tokenizer = AutoTokenizer.from_pretrained(
            "Hate-speech-CNERG/dehatebert-mono-english")
        self.threshold = threshold

        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device('cpu')

        self._model.to(self._device)
        self._softmax = torch.nn.Softmax(dim=1)

    def query(self, doc):
        """Runs all sentences across cores."""
        labels = np.zeros(len(self.labels))  # Per sentence counts of labels.
        scores = np.zeros(len(self.labels))  # Mean score per label.

        # Preprocess sentences (filter, add delimiters, tokenize).
        sentences = [s for s in doc.sentences if len(s) > self.threshold]
        sentences = ["[CLS] " + s + " [SEP]" for s in sentences]
        sentences = [self._tokenizer.tokenize(
            s, padding='max_length', truncation=True) for s in sentences]
        sentences = [
            self._tokenizer.convert_tokens_to_ids(s) for s in sentences]
        n = len(sentences)

        # Create attention masks (check if this works correctly)
        attention_masks = []
        for sentence in sentences:
            attention_masks.append([float(s > 0) for s in sentence])

        if n == 0:
            return np.concatenate([labels, scores])

        # TODO: We are most likely going to remove this stuff unless we really
        #       need the dataloader to go fast (unclear).
        #prediction_data = TensorDataset(prediction_inputs, prediction_masks)
        #do we need a sampler? not sure
        #prediction_sampler = SequentialSampler(prediction_data)
        #prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=5, num_workers=1)

        for sentence, mask in zip(sentences, attention_masks):
            # We're squeezing / unsqueezing the batch dim.
            sentence = torch.tensor(sentence).to(self._device).unsqueeze(0)
            mask = torch.tensor(mask).to(self._device).unsqueeze(0)

            with torch.no_grad():
                logits = self._model(
                    sentence, token_type_ids=None, attention_mask=mask)[0]

            softmax = self._softmax(logits).detach().cpu().numpy()
            labels[np.argmax(softmax, axis=1)] += 1  # Increment counter.
            scores += softmax.squeeze()  # Remove batch dim and sum scores.

        scores /= n  # Take the mean.

        return np.concatenate([labels, scores])

# XPLAIN
#    "0": "hate speech",
#    "1": "normal",
#    "2": "offensive"
