import numpy as np


def loadGloveModel(File):
    """
    reads pretrained glove embedding file and put all words and there embeddings in a dictionary
    @param File:
    @return: a dictionary with key as words and their embeddings as value
    """
    # print("Loading Glove Model")
    f = open(File, 'r')
    gloveModel = {}
    for line in f:
        splitLines = line.split()
        word = splitLines[0]
        wordEmbedding = np.array([float(value) for value in splitLines[1:]])
        gloveModel[word] = wordEmbedding
    # print(len(gloveModel)," words loaded!")
    return gloveModel


embedding = loadGloveModel(
    '/home/supriti/Computational Linguistics/Previous/Team lab/ArtistPrediction/benchmark/glove.6B/glove.6B.300d.txt')


class Embedding(object):
    """
    represent each words to a vector of numbers.
    """

    def __init__(self):
        self.list_with_embeddings = []
        self.PAD = np.zeros(300)
        self.UNK = np.random.randn(300)

    def __call__(self, sample):
        list_with_embeddings = []
        text, intent = sample['text'], sample['intent']
        for word in text:
            if word == 'AHANA':
                list_with_embeddings.append(self.PAD)
            elif word not in embedding:
                list_with_embeddings.append(self.UNK)
            else:
                list_with_embeddings.append(embedding[word])
        return list_with_embeddings
