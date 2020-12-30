from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')


class Tokenize_text(object):
    """
    Tokenization of text using:
    - remove punctuations
    - remove single characters.
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        text = sample['text']
        list_text = tokenizer.tokenize(text)
        list_text = [word for word in list_text if len(word) > 1]
        sample['processed_text'] = list_text
        return sample
