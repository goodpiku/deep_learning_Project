import string


class RemovePunctuation(object):
    def __init__(self):
        self.punctuations = set(string.punctuation)

    def __call__(self, sample):
        text = sample['processed_text']
        filtered_text = [word for word in text if word not in self.punctuations]
        sample['processed_text'] = filtered_text
        return sample
