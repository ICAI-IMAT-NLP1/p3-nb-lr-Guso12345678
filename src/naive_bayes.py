import torch
from collections import Counter, defaultdict
from typing import Dict

try:
    from src.utils import SentimentExample
    from src.data_processing import bag_of_words
except ImportError:
    from utils import SentimentExample
    from data_processing import bag_of_words


class NaiveBayes:
    def __init__(self):
        """
        Initializes the Naive Bayes classifier
        """
        self.class_priors: Dict[int, torch.Tensor] = {}
        self.conditional_probabilities: Dict[int, torch.Tensor] = {}
        self.vocab_size: int = None

    def fit(self, features: torch.Tensor, labels: torch.Tensor, delta: float = 1.0):
        """
        Trains the Naive Bayes classifier by initializing class priors and estimating conditional probabilities.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.
        """
        # TODO: Estimate class priors and conditional probabilities of the bag of words 
        self.class_priors = self.estimate_class_priors(labels)
        self.vocab_size = features.shape[1] # Shape of the probability tensors, useful for predictions and conditional probabilities
        self.conditional_probabilities = self.estimate_conditional_probabilities(features,labels,delta)
        return

    def estimate_class_priors(self, labels: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Estimates class prior probabilities from the given labels.

        Args:
            labels (torch.Tensor): Labels corresponding to each training example.

        Returns:
            Dict[int, torch.Tensor]: A dictionary mapping class labels to their estimated prior probabilities.
        """
        # TODO: Count number of samples for each output class and divide by total of samples
        labels_nueva = labels.clone()
        total_labels = len(labels_nueva)
        label_primera = len(labels_nueva[labels_nueva == 0])
        label_segunda = len(labels_nueva[labels_nueva == 1])
        class_priors: Dict[int, torch.Tensor] = {}
        class_priors[0] = torch.tensor(label_primera/total_labels)
        class_priors[1] = torch.tensor(label_segunda/total_labels)
        return class_priors

    def estimate_conditional_probabilities(
        self, features: torch.Tensor, labels: torch.Tensor, delta: float
    ) -> Dict[int, torch.Tensor]:
        """
        Estima las probabilidades condicionales P(w | y) usando Laplace Smoothing.

        Args:
            features (torch.Tensor): Representación Bag of Words de los ejemplos de entrenamiento.
            labels (torch.Tensor): Etiquetas de los ejemplos de entrenamiento.
            delta (float): Parámetro de suavizado de Laplace.

        Returns:
            Dict[int, torch.Tensor]: Probabilidades condicionales de cada palabra para cada clase.
        """
        vocab_size = features.shape[1]
        palabras_clase = {}  
        palabras_totales_clase = {}  
        for i in range(len(features)):
            etiquetas = labels[i].item()
            if etiquetas not in palabras_clase:
                palabras_clase[etiquetas] = torch.zeros(vocab_size, dtype=torch.float32)  
                palabras_totales_clase[etiquetas] = 0  

            palabras_clase[etiquetas] += features[i]  
            palabras_totales_clase[etiquetas] += features[i].sum().item()  
        probabilidades_clase = {}
        for class_label, word_counts in palabras_clase.items():
            total_words_in_class = palabras_totales_clase[class_label]  
            probabilidades_clase[class_label] = (word_counts + delta) / (total_words_in_class + delta * vocab_size)
        return probabilidades_clase

    def estimate_class_posteriors(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate the class posteriors for a given feature using the Naive Bayes logic.

        Args:
            feature (torch.Tensor): The bag of words vector for a single example.

        Returns:
            torch.Tensor: Log posterior probabilities for each class.
        """
        if self.conditional_probabilities is None or self.class_priors is None:
            raise ValueError(
                "Model must be trained before estimating class posteriors."
            )
        # TODO: Calculate posterior based on priors and conditional probabilities of the words
        tensor_vacio = []
        for i in range(len(self.class_priors)): 
            tensor_log_probailidades = torch.log(self.class_priors[i]) + torch.sum(feature*torch.log(self.conditional_probabilities[i]))
            tensor_vacio.append(tensor_log_probailidades)
        return torch.tensor(tensor_vacio)


    def predict(self, feature: torch.Tensor) -> int:
        """
        Classifies a new feature using the trained Naive Bayes classifier.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example to classify.

        Returns:
            int: The predicted class label (0 or 1 in binary classification).

        Raises:
            Exception: If the model has not been trained before calling this method.
        """
        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")
        
        # TODO: Calculate log posteriors and obtain the class of maximum likelihood 
        pred: int = torch.argmax(self.estimate_class_posteriors(feature))
        return int(pred)

    def predict_proba(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Predict the probability distribution over classes for a given feature vector.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example.

        Returns:
            torch.Tensor: A tensor representing the probability distribution over all classes.

        Raises:
            Exception: If the model has not been trained before calling this method.
        """
        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")

        # TODO: Calculate log posteriors and transform them to probabilities (softmax)
        probs: torch.Tensor = torch.nn.functional.softmax(self.estimate_class_posteriors(feature))
        return probs
