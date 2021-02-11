import json
import random
import os
import numpy as np
import torch


def process_data(filepath, training):
    label_file = '../benchmark/labels.json'
    dict_of_intent_labels = {}
    if os.path.exists(label_file):
        with open(label_file, 'r') as f_file:
            dict_of_intent_labels = json.load(f_file)

    slot_file = '../benchmark/slot_labels.json'
    dict_of_slot_labels = {}
    if os.path.exists(slot_file):
        with open(slot_file, 'r') as r_file:
            dict_of_slot_labels = json.load(r_file)

    if training:
        with open(filepath)as json_file:
            data = json.load(json_file)
        # keys = list(data.keys())
        # random.shuffle(keys)
        if not os.path.exists(label_file):
            list_of_intent = []
            for key in data:
                list_of_intent.append(data[key]['intent'])
            label_of_intent = list(set(list_of_intent))
            print(label_of_intent)
            for label in label_of_intent:
                dict_of_intent_labels[label] = len(dict_of_intent_labels)
            with open('../benchmark/labels.json', 'w') as out_file:
                json.dump(dict_of_intent_labels, out_file)

        # creating slot labels
        if not os.path.exists(slot_file):
            list_of_slots = []
            for key, value in data.items():
                # print(value['slots'])
                for slot in value['slots']:
                    # print(slot)
                    b = f'b-{slot}'
                    i = f'i-{slot}'
                    list_of_slots.append(b)
                    list_of_slots.append(i)
            label_of_slot = list(set(list_of_slots))
            label_of_slot.append('o')
            label_of_slot.append('[PAD]')

            for label in label_of_slot:
                dict_of_slot_labels[label] = len(dict_of_slot_labels)
                with open('../benchmark/slot_labels.json', 'w') as out_file:
                    json.dump(dict_of_slot_labels, out_file)

        list_of_dict = []
        for _, text_and_intent in data.items():
            dict_of_reverse_slots = {}
            for slot, str in text_and_intent['slots'].items():
                words = str.split(' ')
                if len(words) == 1:
                    dict_of_reverse_slots[words[0]] = f'b-{slot}'
                elif len(words) > 1:
                    for word in words:
                        if word == words[0]:
                            dict_of_reverse_slots[word] = f'b-{slot}'
                        else:
                            dict_of_reverse_slots[word] = f'i-{slot}'
            list_of_slots = []
            words = text_and_intent['text'].split(' ')
            for word in words:
                if word in dict_of_reverse_slots:
                    list_of_slots.append(dict_of_reverse_slots[word])
                else:
                    list_of_slots.append('o')

            # slot_tensor= torch.tensor(list_of_slot_labels, dtype=torch.float)
            list_of_dict.append({"text": text_and_intent["text"], "intent": text_and_intent["intent"],
                                 'slots': list_of_slots})
        # print(list_of_dict[0])

    else:
        with open(filepath)as json_file:
            data = json.load(json_file)
            list_of_dict = []
            keys = list(data.keys())
            for key in keys:
                new_data = {'text': data[key]['text']}
                list_of_dict.insert(int(key), new_data)

    return list_of_dict


def foo(sample, slot_to_index):
    if 'slots' not in sample:
        return sample
    slots = sample['slots']
    list_of_slot_labels = []
    words = sample['text'].split(' ')
    for word in words:
        list_of_word = np.zeros(len(slot_to_index))
        if word in slots:
            key = slots[word]
            val = slot_to_index[key]
        elif word == 'AHANA':
            val = slot_to_index['[PAD]']
        else:
            val = slot_to_index['o']
        list_of_word[val] = 1
        list_of_slot_labels.append(list_of_word)

    sample['slots'] = torch.tensor(list_of_slot_labels, dtype=torch.float)
    # print(slot_to_index)
    # y=torch.tensor(sample['slots'], dtype=torch.float)
    return sample


if __name__ == '__main__':
    training = True
    p = process_data('../benchmark/train.json', training)
    print(p)
    # with open('../benchmark/slot_labels.json', 'r') as s_file:
    #     slot_to_index = json.load(s_file)
    # print(len(slot_to_index))
    # print(foo(p, slot_to_index))
