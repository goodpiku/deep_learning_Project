_PAD_ = 'AHANA'


class Resize(object):
    """Rescaling the sample texts to have a fixed length.
    If the length of the text is more than the fixed length then we truncate the text.
    otherwise pad new strings to the text
    """

    def __init__(self, length):
        self.text_length = length

    def __call__(self, sample):
        text, intent = sample['text'], sample['intent']
        list_text = text.split(' ')
        if len(list_text) >= self.text_length:
            list_text = list_text[:self.text_length + 1]
        else:
            while len(list_text)<10:
                list_text.append(_PAD_)
        reformed_text = ' '.join(list_text)

        return {'text': reformed_text, 'intent': intent}
