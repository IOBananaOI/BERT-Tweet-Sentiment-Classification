import pandas as pd

import torch 

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder

from transformers import BertTokenizer
from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents

class TweetsDataset(Dataset):
    def __init__(self, data_path, tokenizer) -> None:
        super().__init__()
        
        self.tokenizer = tokenizer
        
        self.data = pd.read_csv(data_path, encoding='latin-1')
        
        self.data = self.data[["OriginalTweet", "Sentiment"]]
        
        le = LabelEncoder()
        
        self.data["Sentiment"] = le.fit_transform(self.data["Sentiment"])
        
    def __getitem__(self, idx):
        data = self.data.iloc[idx].values
        
        data[0] = self.tokenizer.encode(data[0], return_tensors="pt", padding='max_length', max_length=248)
        
        return list(data)
    
    def __len__(self):
        return len(self.data)
    

def get_datasets(train_data_path, test_data_path):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", model_max_length=248)
    
    # normalizer = normalizers.Sequence([NFD(), StripAccents()])
    
    # tokenizer.normalizer = normalizer
    
    train_ds = TweetsDataset(train_data_path, tokenizer)
    
    if test_data_path is not None:
        test_ds = TweetsDataset(test_data_path, tokenizer)
        
        return train_ds, test_ds

    return train_ds, None

def get_dataloaders(train_data_path, test_data_path):
    
    train_ds, test_ds = get_datasets(train_data_path, test_data_path)
    
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
    
    test_dl = DataLoader(test_ds, batch_size=16, shuffle=False)
    
    return train_dl, test_dl