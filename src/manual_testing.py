import json
import random
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def evaluation(actual_labels, predicted_labels):  # test this
    micro_precision = precision_score(actual_labels, predicted_labels, average='micro', zero_division='warn')
    micro_recall = recall_score(actual_labels, predicted_labels, average='micro', zero_division='warn')
    micro_f1_score = f1_score(actual_labels, predicted_labels, average='micro', zero_division='warn')

    macro_precision = precision_score(actual_labels, predicted_labels, average='macro', zero_division='warn')
    macro_recall = recall_score(actual_labels, predicted_labels, average='macro', zero_division='warn')
    macro_f1_score = f1_score(actual_labels, predicted_labels, average='macro', zero_division='warn')
    dict_of_evaluation = {'micro_p': micro_precision, 'macro_p': macro_precision, 'micro_r': micro_recall,
                          'macro_r': macro_recall, 'micro_f1': micro_f1_score, 'macro_f1': macro_f1_score}

    return dict_of_evaluation


num_to_manually_verify = 100

dev_output_file_path = '../benchmark/dev_output.json'
dev_file = '../benchmark/dev.json'

with open('../benchmark/labels.json', 'r') as j_file:
    intent_to_index = json.load(j_file)

with open(dev_output_file_path, 'r') as f:
    dev_output = json.load(f)

with open(dev_file, 'r') as f:
    dev = json.load(f)

prediction_labels = []
manual_labels = []

num_of_dev_output = len(dev_output)
keys = list(dev_output.keys())
random.shuffle(keys)

# for key in keys:
#     print(key)
#     if dev[key]['text'] != dev_output[key]['text']:
#         print('N')
#     else:
#         print('-')

keys = keys[:num_to_manually_verify]

for key in keys:
    predictions = dev_output[key]
    text = predictions['text']
    predicted_label = int(intent_to_index[predictions['intent']])
    print(f'\n num. {num_to_manually_verify} -- key={key}, {intent_to_index}\nText -> {text}, intent=?')
    manual_input = int(input())

    manual_labels.append(manual_input)
    prediction_labels.append(predicted_label)
    if predicted_label == manual_input:
        print('Prediction matches manual')
    else:
        print(f'Prediction "{predictions["intent"]}" does not match manual')
    print('-' * 40)
    num_to_manually_verify -= 1

print(evaluation(predicted_labels=prediction_labels, actual_labels=manual_labels))
