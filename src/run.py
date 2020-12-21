import json
from dataset import Intent_identification_dataset
from data_transformation.tokenize import Tokenize_text
from data_transformation.resize import Resize
from data_transformation.lematization import Lemmatize_text
from data_transformation.glove_embedding import loadGloveModel
from data_transformation.embedding import Embedding
from data_transformation.label_transformation import Label
from torchvision import transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from models.Intent_classifier_RNN import IntentclassifierRNN
from models.Intent_classifier_CNN import IntentclassifierCNN
import torch
from data_processing import process_data
from typing import List, Dict

device = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu')
model_name = 'RNN'
batch_size = 56
num_epochs = 100
num_of_words = 20
embedding_size = 300
embedding_file_path = '../benchmark/glove.6B.300d.txt'

embedding_model = loadGloveModel(embedding_file_path)

# with open('../benchmark/less_train.json')as json_file:
#     data = json.load(json_file)
training = True
data = process_data('../benchmark/train.json', training)

with open('../benchmark/labels.json', 'r') as j_file:
    intent_to_index = json.load(j_file)

""" below are the Data Processing steps including tokenization, lemmatization, resizing, embedding etc"""
token = Tokenize_text()
lemma = Lemmatize_text()
resize = Resize(num_of_words)
embedding = Embedding(embedding_model)
label_index = Label(intent_to_index)
compose = transforms.Compose([token, lemma, resize, embedding, label_index])
dataset = Intent_identification_dataset(data, compose)
# print(dataset[0])

""" Split data into training and validation"""
len_of_dataset = len(dataset)
tr_size = int(len_of_dataset * 0.8)  # training data
vl_size = len_of_dataset - tr_size
train, validation = random_split(dataset, [tr_size, vl_size])
# print(len(train), len(validation))

"""Creating batches"""
training_dataset_with_batches = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True)
validation_dataset_with_batches = DataLoader(validation, batch_size=batch_size, shuffle=True, drop_last=True)
# print(len(training_dataset_with_batches), len(validation_dataset_with_batches))

if model_name == 'RNN':
    model = IntentclassifierRNN(embedding_size=embedding_size, num_of_labels=len(intent_to_index))
elif model_name == 'CNN':
    model = IntentclassifierCNN(embedding_size=embedding_size, num_of_labels=len(intent_to_index))
else:
    pass

model.to(
    device)  # I do his here to avoid writing this in every classifier file. model.to(device means moving the model from cpu to gpu
result = model.run_epochs(training_data=training_dataset_with_batches, validation_data=validation_dataset_with_batches,
                          num_epochs=num_epochs)
# print(result)
