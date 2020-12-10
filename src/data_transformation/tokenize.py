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
        self.list_text = []

    def __call__(self, sample):
        text, intent = sample['text'], sample['intent']
        # self.list_text = word_tokenize(text)
        self.list_text = tokenizer.tokenize(text)
        self.list_text = [word for word in self.list_text if len(word) > 1]

        return {'text': self.list_text, 'intent': intent}
