from typing import List, Dict
from collections import Counter
import torch
import numpy as np 

try:
    from src.utils import SentimentExample, tokenize
except ImportError:
    from utils import SentimentExample, tokenize


import numpy as np
from typing import List

import numpy as np
from typing import List

import numpy as np
from typing import List

def read_sentiment_examples(infile: str) -> List[SentimentExample]:
    """
    Reads sentiment examples from a file.

    Args:
        infile: Path to the file to read from.

    Returns:
        A list of SentimentExample objects parsed from the file.
    """
    examples: List[SentimentExample] = []

    with open(infile, "r", encoding="utf-8") as file:
        lines = file.read().splitlines()  # Leer líneas del archivo

    for line in lines:
        try:
            line = line.strip()
            partes = line.rsplit("\t", 1)
            texto, label_str = partes
            label = int(label_str)
            palabra_tokenizada = tokenize(texto)
            examples.append(SentimentExample(palabra_tokenizada, label))

        except: 
            continue
    return examples
def build_vocab(examples: List[SentimentExample]) -> Dict[str, int]:
    """
    Creates a vocabulary from a list of SentimentExample objects.

    The vocabulary is a dictionary where keys are unique words from the examples and values are their corresponding indices.

    Args:
        examples (List[SentimentExample]): A list of SentimentExample objects.

    Returns:
        Dict[str, int]: A dictionary representing the vocabulary, where each word is mapped to a unique index.
    """
    # TODO: Count unique words in all the examples from the training set
    vocab: Dict[str, int] = {}
    indice = 0 
    for palabra in examples: 
        for p in palabra.words: 
            if p not in vocab.keys(): 
                vocab[p] = indice 
                indice += 1
    return vocab


def bag_of_words(
    text: List[str], vocab: Dict[str, int], binary: bool = False
) -> torch.Tensor:
    """
    Converts a list of words into a bag-of-words vector based on the provided vocabulary.
    Supports both binary and full (frequency-based) bag-of-words representations.

    Args:
        text (List[str]): A list of words to be vectorized.
        vocab (Dict[str, int]): A dictionary representing the vocabulary with words as keys and indices as values.
        binary (bool): If True, use binary BoW representation; otherwise, use full BoW representation.

    Returns:
        torch.Tensor: A tensor representing the bag-of-words vector.
    """
    # TODO: Converts list of words into BoW, take into account the binary vs full
    if binary == True:
        diccionario = {}
        for palabra,valor in vocab.items(): 
            if palabra in text: 
                diccionario[palabra] = 1
            else: 
                diccionario[palabra] = 0
        lista_valores = [valor for palabra,valor in diccionario.items()] 
        bow: torch.Tensor = torch.tensor(lista_valores)
        return bow
    else:
        lista = []
        for palabra,valor in vocab.items(): 
            lista.append(text.count(palabra))
        bow:torch.Tensor = torch.tensor(lista)
        return bow
