_PAD_ = 'AHANA'


class Resize(object):
    """Resizing the sample texts to have a fixed length.
    If the length of the text is more than the fixed length then we truncate the text.
    otherwise pad new strings to the text
    """

    def __init__(self, length):
        self.text_length = length

    def __call__(self, sample):
        text, intent = sample['text'], sample['intent']

        if len(text) >= self.text_length:  # if length of text more than 10
            text = text[:self.text_length + 1]
        else:
            while len(text) < 10:  # if length of text less than 10.
                text.append(_PAD_)

        return {'text': text, 'intent': intent}
