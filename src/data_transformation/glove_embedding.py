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


