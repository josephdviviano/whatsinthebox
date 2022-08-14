#from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from hatesonar import Sonar
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing
import time
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import os
import multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
#from cc_net import dedup, execution, jsonql, minify, perplexity, process_wet_file,  text_normalizer
from witb.utils.textutils import normalize_line
import kenlm  # type: ignore
import sentencepiece  # type: ignore
from copy import copy


#class PerplexityRunner():
#    def __init__():


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
            if result['top_class'] == 'hate_speech':
                print("SONAR ", sentence)
            # Gets the numeric score for each class per sentence.
            for r in result['classes']:
                scores[self.labels[r['class_name']]] += r['confidence']


        scores /= n  # Take the mean.

        return np.concatenate([labels, scores])


class DeLimitRunner():
    def __init__(self, threshold=20, max_sentences=30):
        self._model = AutoModelForSequenceClassification.from_pretrained(
            "Hate-speech-CNERG/dehatebert-mono-english")
        self.labels = {
            'hate_speech': 0,
            'normal': 1}
        self._tokenizer = AutoTokenizer.from_pretrained(
            "Hate-speech-CNERG/dehatebert-mono-english")
        self.threshold = threshold
        self.max_sentences = int(max_sentences)
        self._n_cpu = multiprocessing.cpu_count()

        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device('cpu')

        self._model.to(self._device)
        self._softmax = torch.nn.Softmax(dim=1)

    def query(self, doc):
        """Runs all sentences across cores."""
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        labels = np.zeros(len(self.labels))  # Per sentence counts of labels.
        scores = np.zeros(len(self.labels))  # Mean score per label.

        # Preprocess sentences (filter, add delimiters, tokenize).
        # Sentences must be > 20 characters.
        sentences = [s for s in doc.sentences if len(s) > self.threshold]

        # Do the first n sentences only so we cap runtime.
        if len(sentences) > self.max_sentences:
            sentences = sentences[:self.max_sentences]

        raw_sentences = copy(sentences)
        sentences = ["[CLS] " + s + " [SEP]" for s in sentences]
        sentencetexts=[s for s in sentences]
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

        dataset = TensorDataset(torch.LongTensor(sentences),
                                torch.LongTensor(attention_masks))
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset,
                                sampler=sampler,
                                batch_size=self.max_sentences,
                                num_workers=1)

        for i, batch in enumerate(dataloader):

            sentence, mask = batch
            sentence = sentence.to(self._device)
            mask = mask.to(self._device)


            with torch.no_grad():
                logits = self._model(
                    sentence, token_type_ids=None, attention_mask=mask)[0]

            softmax = self._softmax(logits).detach().cpu().numpy()

            # Count sentences with each label.
            idx = np.argmax(softmax, axis=1)
            _labels = np.zeros(softmax.shape)
            _labels[np.arange(_labels.shape[0]), idx] = 1
            labels += _labels.sum(0)

            # Get the hateful/nice sentences:
            batch_size = sentence.shape[0]
            batch_idx = np.arange(batch_size*i, batch_size*(i+1))

            hate_idx = batch_idx[np.where(idx == 1)[0]]
            nice_idx = batch_idx[np.where(idx == 0)[0]]

            if len(hate_idx) > 0:
                print("DELIMIT HATE:", np.array(raw_sentences)[hate_idx])

            #print("DELIMIT NICE:", np.array(raw_sentences)[nice_idx])
            #print('\n')

            scores += softmax.sum(0)  # Sum scores over batch dimension.

        scores /= n  # Take the mean.

        return np.concatenate([labels, scores])


class PerplexRunner():
    def __init__(self, threshold=20):
        #TODO: generalize these paths.
        self.sp_model = sentencepiece.SentencePieceProcessor(
            '/home/mila/l/lucciona/cc_net/data/lm_sp/en.sp.model')
        self._model= kenlm.Model(
            '/home/mila/l/lucciona/cc_net/data/lm_sp/en.arpa.bin')
        self.threshold = threshold

    def pp(log_score, length):
        return

    def query(self, doc):
        """Runs all sentences across cores."""

        # Remove short sentences.
        sentences = [s for s in doc.sentences if len(s) > self.threshold]
        n = len(sentences)
        score= 0.000

        if n == 0:
            return -np.inf  # Worst possible perplexity.

        log_score, doc_length = 0, 0

        for sentence in sentences:
            sentence = normalize_line(sentence)
            sentence = self.sp_model .encode_as_pieces(sentence)
            log_score += self._model.score(" ".join(sentence))
            doc_length += len(sentence) + 1

        #score = (10.0 ** (-log_score / doc_length))
        score = round(10.0 ** (-log_score/doc_length), 1)
        return score
