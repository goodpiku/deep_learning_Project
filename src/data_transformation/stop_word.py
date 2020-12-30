from nltk.corpus import stopwords
import nltk

stop_words = set(stopwords.words('english'))


# words = set(nltk.corpus.words.words())


class Stopword(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        text = sample['processed_text']
        filtered_text = [word for word in text if not word in stop_words]
        # joined_filtered_text=" ".join(filtered_text)
        # mis_w = [w for w in nltk.wordpunct_tokenize(joined_filtered_text) if w.lower() in words or not w.isalpha()]
        sample['processed_text'] = filtered_text
        return sample
