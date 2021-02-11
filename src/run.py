import json
from dataset import Intent_identification_dataset
from data_transformation.tokenize import Tokenize_text
from data_transformation.resize import Resize
from data_transformation.lematization import Lemmatize_text
from data_transformation.glove_embedding import loadGloveModel
from data_transformation.embedding import Embedding
from data_transformation.label_transformation import Label
from data_transformation.slot_transformation import Slot
from data_transformation.stop_word import Stopword
from data_transformation.remove_punctuations import RemovePunctuation
from torchvision import transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from models.Intent_classifier_RNN import IntentclassifierRNN
from models.Intent_classifier_CNN import IntentclassifierCNN
import torch
from data_processing import process_data
import gensim
import fasttext
from typing import List, Dict

device = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu')
model_name = 'RNN'
torch.manual_seed(20)
batch_size = 5
num_epochs = 25
num_of_words = 45

training = False
if training:
    data = process_data('../benchmark/train.json', training)

embedding_size = 300
embedding_file_path = '../benchmark/glove.6B.300d.txt'
# embedding_file_path = '../benchmark/cc.en.300.bin'

embedding_model = loadGloveModel(embedding_file_path)
# embedding_model = fasttext.load_model(embedding_file_path)
# embedding_model = gensim.models.KeyedVectors.load_word2vec_format('../benchmark/wiki.en.vec', binary=False)


with open('../benchmark/labels.json', 'r') as j_file:
    intent_to_index = json.load(j_file)
with open('../benchmark/slot_labels.json', 'r') as s_file:
    slot_to_index = json.load(s_file)

if model_name == 'RNN':
    model = IntentclassifierRNN(embedding_size=embedding_size, num_of_labels=len(intent_to_index),
                                num_of_slots=len(slot_to_index))
elif model_name == 'CNN':
    model = IntentclassifierCNN(embedding_size=embedding_size, num_of_labels=len(intent_to_index),
                                num_of_slots=len(slot_to_index))
else:
    pass
model.to(
    device)  # I do this here to avoid writing this in every classifier file. model.to(device means moving the model from cpu to gpu

""" below are the Data Processing steps including tokenization, lemmatization, resizing, embedding etc"""
token = Tokenize_text()
# ohne_stop = Stopword()
no_punctions = RemovePunctuation()
lemma = Lemmatize_text()
resize = Resize(num_of_words)
slot_index = Slot(slot_to_index)
embedding = Embedding(embedding=embedding_model, embedding_dim=embedding_size)
label_index = Label(intent_to_index)

if training:
    list_of_transforms = [token, resize, label_index, slot_index, embedding]
    """ Split data into training and validation"""
    compose = transforms.Compose(list_of_transforms)
    dataset = Intent_identification_dataset(data, compose)
    len_of_dataset = len(dataset)
    print(f'length of dataset : {len_of_dataset}')
    tr_size = int(len_of_dataset * 0.8)  # training data
    vl_size = len_of_dataset - tr_size
    train, validation = random_split(dataset, [tr_size, vl_size])
    # print(len(train), len(validation))
    """Creating batches"""
    training_dataset_with_batches = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=False)
    validation_dataset_with_batches = DataLoader(validation, batch_size=batch_size, shuffle=True, drop_last=False)
    print(
        f'length of training data is  {len(train)} and length of validation data is {len(validation)}')
    test_text, predictions = model.run_epochs(training_data=training_dataset_with_batches,
                                              validation_data=validation_dataset_with_batches,
                                              num_epochs=num_epochs)

if training == False:
    """
    """
    list_of_transforms = [token, resize, embedding]
    data = process_data('../benchmark/dev.json', training=False)
    compose = transforms.Compose(list_of_transforms)
    dataset = Intent_identification_dataset(data, compose)
    test_dataset_with_batches = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    model.load_model(path_to_saved_model='../benchmark/results/parameters/parameters_0.4955923172855647.pt')
    test_text, predictions_1, predictions_2 = model.run_test(test_data=test_dataset_with_batches)
    label_to_intent = {}
    for key, value in intent_to_index.items():
        label_to_intent[value] = key
    label_to_slot = {}
    for key, val in slot_to_index.items():
        label_to_slot[val] = key
    predictions_with_labels = {}
    predictions_for_upload = {}
    for pos, label in enumerate(predictions_1):
        predictions_with_labels[pos] = {'text': test_text[pos], 'intent': label_to_intent[label]}
        predictions_for_upload[pos] = {'intent': label_to_intent[label]}

    """
    Todo:
    label_to_slot ={0: 'b-city', 1: 'b-sort', 2: 'b-service', 3: 'i-rating_value', 4: 'i-restaurant_type', 5: 'i-condition_description', ...........
                    73: 'i-sort', 74: 'b-music_item', 75: 'b-spatial_relation', 76: 'b-current_location', 77: 'b-genre', 78: 'o', 79: '[PAD]'}
                    
    predictions_2 = [[78, 78, 78, 78, 78, 78, 78, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79], 
    [78, 78, 78, 78, 78, 78, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79],
     [78, 78, 78, 78, 78, 78, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79], ...............
    """
    predictions_with_slots = {}
    for ind, sentence in enumerate(predictions_2):
        text = test_text[ind].split(' ')
        # predictions_with_slots = {'text': test_text[ind], 'slot': {}}
        each_prediction = {}
        slot = {}
        for pos, slot_key in enumerate(sentence[:len(text)]):
            if (label_to_slot[slot_key].find('b-') != -1):
                slot[label_to_slot[slot_key][2:]] = text[pos]
            elif (label_to_slot[slot_key].find('i-') != -1):
                if label_to_slot[slot_key][2:] not in slot:
                    slot[label_to_slot[slot_key][2:]] = text[pos]
                else:
                    slot[label_to_slot[slot_key][2:]] = f'{slot[label_to_slot[slot_key][2:]]} {text[pos]}'
        intent_label = predictions_1[ind]
        predictions_with_slots[ind] = {'text': test_text[ind], 'intent': label_to_intent[intent_label], 'slots': slot}
    # print(predictions_with_slots)

    with open('../benchmark/dev_output.json', 'w') as out_file:
        json.dump(predictions_with_slots, out_file)
    with open('../benchmark/dev_upload.json', 'w') as out_file:
        json.dump(predictions_for_upload, out_file)

