import json
from dataset import Intent_identification_dataset
from resize import Resize
from torchvision import transforms
from torch.utils.data import random_split
from typing import List, Dict

with open('../benchmark/less_train.json')as json_file:
    data = json.load(json_file)
resize = Resize(10)
compose = transforms.Compose([resize])
dataset = Intent_identification_dataset(data, compose)
print(dataset[0])

len_of_dataset = len(dataset)
tr_size = int(len_of_dataset * 0.8)  # 80% training data
vl_size = len_of_dataset - tr_size
train, validation = random_split(dataset, [tr_size, vl_size])
