from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


class Lemmatize_text(object):
    """
     - make all characters of string to lower
     - replace the word with the lemma.
    """

    def __init__(self):
        self.lemmatized_text = []

    def __call__(self, sample):
        text, intent = sample['text'], sample['intent']
        self.lemmatized_text = []
        for token in text:
            token = token.lower()
            self.lemmatized_text.append(lemmatizer.lemmatize(token))

        return {'text': self.lemmatized_text, 'intent': intent}