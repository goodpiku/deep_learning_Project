import json
import random
import os


def process_data(filepath, training):
    label_file = '../benchmark/labels.json'
    dict_of_intent_labels = {}
    if os.path.exists(label_file):
        with open(label_file, 'r') as f_file:
            dict_of_intent_labels = json.load(f_file)

    if training:
        with open(filepath)as json_file:
            data = json.load(json_file)
        keys = list(data.keys())
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
        list_of_dict = []
        for _, text_and_intent in data.items():
            list_of_dict.append({"text": text_and_intent["text"], "intent": text_and_intent["intent"]})
    else:
        with open(filepath)as json_file:
            data = json.load(json_file)
            list_of_dict = []
            keys = list(data.keys())
            for key in keys:
                new_data = {'text': data[key]['text']}
                list_of_dict.insert(int(key), new_data)

    return list_of_dict
