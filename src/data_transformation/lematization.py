from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


class Lemmatize_text(object):
    """
     - make all characters of string to lower
     - replace the word with the lemma.
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        lemmatized_text = []
        text = sample['processed_text']
        for token in text:
            token = token.lower()
            lemmatized_text.append(lemmatizer.lemmatize(token))
        sample['processed_text'] = lemmatized_text
        return sample
