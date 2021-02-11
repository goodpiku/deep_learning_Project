import torch
import numpy as np


class Slot:

    def __init__(self, dict_of_slot_to_index):
        self.slot_to_index = dict_of_slot_to_index

    def __call__(self, sample):
        if 'slots' not in sample:
            return sample
        slots = sample['slots']
        list_of_slot_labels = []
        for word in sample['processed_text']:
            list_of_word = np.zeros(len(self.slot_to_index))
            if word in slots:
                key = slots[word]
                val = self.slot_to_index[key]
            elif word == 'AHANA':
                val = self.slot_to_index['[PAD]']
            else:
                val = self.slot_to_index['o']
            list_of_word[val] = 1
            list_of_slot_labels.append(list_of_word)

        sample['slot_label'] = torch.tensor(list_of_slot_labels, dtype=torch.float)
        return sample
