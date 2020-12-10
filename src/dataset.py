import json
from torch.utils.data import Dataset, DataLoader

class Intent_identification_dataset(Dataset):
    """
    Take the whole dataset and
    return instances of the dataset.
    """

    def __init__(self, data, transform):
        self.dataset = data
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if self.transform:
            sample=self.transform(sample)
        return sample


if __name__ == '__main__':
    with open('../benchmark/less_train.json')as json_file:
        data = json.load(json_file)
    dataset = Intent_identification_dataset(data)
    print(dataset[12])
    print(dataset[1])
