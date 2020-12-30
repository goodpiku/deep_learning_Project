import numpy as np
import torch

np.random.seed(42)


class Embedding:
    """
    represent each words to a vector of numbers.
    """

    def __init__(self, embedding, embedding_dim):
        self.embedding = embedding
        self.list_with_embeddings = []
        self.PAD = np.zeros(embedding_dim, dtype='float32')
        self.UNK = np.random.uniform(-0.25, 0.25, embedding_dim)

    def __call__(self, sample):
        list_with_embeddings = []
        text = sample['processed_text']
        for word in text:
            if word == 'AHANA':
                list_with_embeddings.append(self.PAD)
            elif word not in self.embedding:
                list_with_embeddings.append(self.UNK)
            else:
                list_with_embeddings.append(self.embedding[word])
        sample['processed_text'] = torch.tensor(list_with_embeddings).float()
        return sample
