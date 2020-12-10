import json
from dataset import Intent_identification_dataset
from tokenize import Tokenize_text
from resize import Resize
from lematization import Lemmatize_text
from torchvision import transforms
from torch.utils.data import random_split
from typing import List, Dict

with open('../../benchmark/less_train.json')as json_file:
    data = json.load(json_file)
token = Tokenize_text()
resize = Resize(10)
lemma = Lemmatize_text(10)
compose = transforms.Compose([token])
# compose = transforms.Compose([resize, lemma])
dataset = Intent_identification_dataset(data, compose)
print(dataset[6])

len_of_dataset = len(dataset)
tr_size = int(len_of_dataset * 0.8)  # 80% training data
vl_size = len_of_dataset - tr_size
train, validation = random_split(dataset, [tr_size, vl_size])
