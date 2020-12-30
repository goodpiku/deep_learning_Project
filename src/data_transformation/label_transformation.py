import torch


class Label:

    def __init__(self, dict_of_intent_to_index):
        self.intent_to_index = dict_of_intent_to_index

    def __call__(self, sample):
        if 'intent' not in sample:
            return sample
        intent = sample['intent']
        intent_label = self.intent_to_index[intent]
        sample['intent_label'] = intent_label
        return sample
