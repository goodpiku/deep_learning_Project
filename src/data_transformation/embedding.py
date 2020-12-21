import numpy as np
import torch


class Embedding:
    """
    represent each words to a vector of numbers.
    """

    def __init__(self, embedding):
        self.embedding = embedding
        self.list_with_embeddings = []
        self.PAD = np.zeros(300)
        self.UNK = np.random.randn(300)

    def __call__(self, sample):
        list_with_embeddings = []
        text, intent = sample['text'], sample['intent']
        for word in text:
            if word == 'AHANA':
                list_with_embeddings.append(self.PAD)
            elif word not in self.embedding:
                list_with_embeddings.append(self.UNK)
            else:
                list_with_embeddings.append(self.embedding[word])
        return {'text': torch.FloatTensor(list_with_embeddings), 'intent': sample['intent']}
