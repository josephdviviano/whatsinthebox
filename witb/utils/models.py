import numpy as np
from hatesonar import Sonar
import multiprocessing
from multiprocessing import Pool
import torch
import tensorflow as tf
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score

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

class DeLimit_Runner():
    def __init__(self, threshold=20):
        self._model = model = AutoModelForSequenceClassification.from_pretrained("Hate-speech-CNERG/dehatebert-mono-english")
        self.labels = {
            'hate_speech': 0,
            'normal': 1}
        self.tokenizer = AutoTokenizer.from_pretrained("Hate-speech-CNERG/dehatebert-mono-english")
        self.threshold = threshold

    def proc_sentence(self, sentence):
        """Results for a single sentence."""
        labels = np.zeros(len(self.labels))
        scores = np.zeros(len(self.labels))


        return (labels, scores)

    def query(self, doc):
        """Runs all sentences across cores."""
        if torch.cuda.is_available():    
            device = torch.device("cuda")
                   
        #TODO - not sure how to do this
        labels = np.zeros(len(self.labels))  # Per sentence counts of labels.
        scores = np.zeros(len(self.labels))  # Mean score per label.

        # Same as Sonar
        sentences = [s for s in doc.sentences if len(s) > self.threshold]
        n = len(sentences)

        if n == 0:
            return np.concatenate([labels, scores])
        
        #New stuff
        #I'm assigning a 'non-hate' label to each sentence as a proxy
        labels= [1] * len(sentences)
        sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]
        tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
        input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
        input_ids = pad_sequences(input_ids, maxlen=20, dtype="long", truncating="post", padding="post")
        # Create attention masks (check if this works correctly)
        attention_masks = [float(i>0) for i in input_ids]
        
        prediction_inputs = torch.tensor(input_ids)
        prediction_masks = torch.tensor(attention_masks)
        prediction_labels = torch.tensor(labels)
        prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
        #do we need a sampler? not sure
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=5, num_workers=1)
            
        predictions , true_labels, pred_labels, eval_accuracy = [], [], [], []
        # Predict 
        for batch in prediction_dataloader:
          # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            model=model.to(device)
          # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
          # Telling the model not to compute or store gradients, saving memory and speeding up prediction
            with torch.no_grad():
            # Forward pass, calculate logit predictions
                logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)[0]

          # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            #tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            # Accumulate the total accuracy.
            #eval_accuracy += tmp_eval_accuracy

            pred_labels+=list(np.argmax(logits, axis=1).flatten())
     
            
            
            
            
            results = logits
            label = pred_labels

            # TODO Gets the numeric score for each class per sentence.
            #for r in result['classes']:
             #   scores[self.labels[r['class_name']]] += r['confidence']

        #scores /= n

        return np.concatenate([labels, results])



# XPLAIN
#    "0": "hate speech",
#    "1": "normal",
#    "2": "offensive"
