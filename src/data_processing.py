import json
import random


def process_data(filepath, training):
    if training:
        with open(filepath)as json_file:
            data = json.load(json_file)
            keys = list(data.keys())
            random.shuffle(keys)
            list_of_intent = []
            for key in data:
                list_of_intent.append(data[key]['intent'])
            label_of_intent = list(set(list_of_intent))
            print(label_of_intent)
            dict_of_intent_labels = {}
            for label in label_of_intent:
                dict_of_intent_labels[label] = len(dict_of_intent_labels)

        list_of_dict = []
        for key in keys:
            new_data = {"text": data[key]["text"], "intent": data[key]["intent"]}
            list_of_dict.append(new_data)
            # with open('../benchmark/less_train.json', 'w')as out_file:
            #     json.dump(list_of_dict, out_file)
            #
            with open('../benchmark/labels.json', 'w') as out_file:
                json.dump(dict_of_intent_labels, out_file)
    else:
        list_of_dict = []
        with open(filepath)as json_file:
            data = json.load(json_file)
            keys = list(data.keys())
            for key in keys:
                new_data = {'text': data[key]['text']}
                list_of_dict.append(new_data)

    return list_of_dict
