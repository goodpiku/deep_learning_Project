_PAD_ = 'AHANA'


class Resize:
    """Resizing the sample texts to have a fixed length.
    If the length of the text is more than the fixed length then we truncate the text.
    otherwise pad new strings to the text
    """

    def __init__(self, length):
        self.text_length = length

    def __call__(self, sample):
        text = sample['processed_text']

        if len(text) > self.text_length:  # if length of text more than 10
            text = text[:self.text_length]
        else:
            while len(text) < self.text_length:  # if length of text less than 10.
                text.append(_PAD_)
        sample['processed_text'] = text
        return sample
