from typing import List, Dict
import torch
import re
import string


def remove_punctuations(input_col):
    """To remove all the punctuations present in the text.Input the text column"""
    table = str.maketrans("", "", string.punctuation)
    return input_col.translate(table)


# Tokenizes a input_string. Takes a input_string (a sentence), splits out punctuation and contractions, and returns a list of
# strings, with each input_string being a token.
def tokenize(input_string):
    input_string = remove_punctuations(input_string)
    input_string = re.sub(r"[^A-Za-z0-9(),.!?\'`\-\"]", " ", input_string)
    input_string = re.sub(r"\'s", " 's", input_string)
    input_string = re.sub(r"\'ve", " 've", input_string)
    input_string = re.sub(r"n\'t", " n't", input_string)
    input_string = re.sub(r"\'re", " 're", input_string)
    input_string = re.sub(r"\'d", " 'd", input_string)
    input_string = re.sub(r"\'ll", " 'll", input_string)
    input_string = re.sub(r"\.", " . ", input_string)
    input_string = re.sub(r",", " , ", input_string)
    input_string = re.sub(r"!", " ! ", input_string)
    input_string = re.sub(r"\?", " ? ", input_string)
    input_string = re.sub(r"\(", " ( ", input_string)
    input_string = re.sub(r"\)", " ) ", input_string)
    input_string = re.sub(r"\-", " - ", input_string)
    input_string = re.sub(r"\"", ' " ', input_string)
    # We may have introduced double spaces, so collapse these down
    input_string = re.sub(r"\s{2,}", " ", input_string)
    return list(filter(lambda x: len(x) > 0, input_string.split(" ")))


class SentimentExample:
    """
    Data wrapper for a single example for sentiment analysis.

    Attributes:
        words (List[str]): List of words.
        label (int): Sentiment label (0 for negative, 1 for positive).
    """

    def __init__(self, words: List[str], label: int):
        self._words = words
        self._label = label

    def __repr__(self) -> str:
        if self.label is not None:
            return f"{self.words}; label={self.label}"
        else:
            return f"{self.words}, no label"

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other) -> bool:
        if not isinstance(other, SentimentExample):
            # don't attempt to compare against unrelated types
            return NotImplemented
        return self.words == other.words and self.label == other.label

    @property
    def words(self):
        return self._words

    @words.setter
    def words(self, value):
        raise NotImplemented

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, value):
        raise NotImplemented


###FUNCIONES AUXILIARES: SE QUE NO ES LO MAS EFICIENTE. 
def verdaderos_positivos(predictions: torch.Tensor, labels: torch.Tensor)->int: 
    mascara_positiva_predictions = (predictions == 1).tolist()
    mascara_positiva_labels = (labels == 1).tolist() 
    cont = 0 
    for i,j in zip(mascara_positiva_predictions,mascara_positiva_labels): 
        if i == j and i == True and j == True: 
            cont += 1 
    return cont
def verdaderos_negativos(predictions: torch.Tensor, labels: torch.Tensor)->int: 
    mascara_positiva_predictions = (predictions == 1).tolist()
    mascara_positiva_labels = (labels == 1).tolist() 
    cont = 0 
    for i,j in zip(mascara_positiva_predictions,mascara_positiva_labels): 
        if i == j and i == False and j == False: 
            cont += 1 
    return cont
def falsos_positivos(predictions: torch.Tensor, labels: torch.Tensor)->int: 
    mascara_positiva_predictions = (predictions == 1).tolist()
    mascara_positiva_labels = (labels == 1).tolist() 
    cont = 0 
    for i,j in zip(mascara_positiva_predictions,mascara_positiva_labels): 
        if i != j and i == True and j == False: 
            cont += 1 
    return cont

def falsos_negativos(predictions: torch.Tensor, labels: torch.Tensor)->int: 
    mascara_positiva_predictions = (predictions == 1).tolist()
    mascara_positiva_labels = (labels == 1).tolist() 
    cont = 0 
    for i,j in zip(mascara_positiva_predictions,mascara_positiva_labels): 
        if i != j and i == False and j == True: 
            cont += 1 
    return cont


def accuracy(predictions: torch.Tensor, labels: torch.Tensor)->float: 
    tp = verdaderos_positivos(predictions,labels)
    tn = verdaderos_negativos(predictions,labels)
    fp = falsos_positivos(predictions,labels)
    fn = falsos_negativos(predictions,labels)
    return (tp+tn)/(tp+tn+fp+fn)

def precision(predictions: torch.Tensor, labels: torch.Tensor)->float: 
    tp = verdaderos_positivos(predictions,labels)
    fp = falsos_positivos(predictions,labels)
    return tp/(tp+fp)

def recall(predictions: torch.Tensor, labels: torch.Tensor)->float: 
    tp = verdaderos_positivos(predictions,labels)
    fn = falsos_negativos(predictions,labels)
    return tp/(tp+fn)

def f1_score(predictions: torch.Tensor, labels: torch.Tensor) ->float: 
    precision1 = precision(predictions,labels)
    recall1 = recall(predictions,labels)
    return (2*precision1*recall1)/(precision1+recall1)

def evaluate_classification(predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """
    Evaluate classification metrics including accuracy, precision, recall, and F1-score.

    Args:
        predictions (torch.Tensor): Predictions from the model.
        labels (torch.Tensor): Actual ground truth labels.

    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    metrics: Dict[str, float] = {"accuracy":accuracy(predictions,labels),"precision":precision(predictions,labels),"recall":recall(predictions,labels),"f1_score":f1_score(predictions,labels)}

    return metrics
