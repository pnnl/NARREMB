'''
Dataset Class
'''

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import csv
import os
import re

class NetworkTwitterDataset(Dataset):
    
    def __init__(self, path, phase="train", sp_thres=4):

        fn = f"{phase}.csv"
            
        path_fn = os.path.join(path, fn)
        
        # source and target - User IDs; source_tweet and target_tweet - Tweet IDs; tweet1 and tweet2 - Tweet Text 
        self.data = pd.read_csv(path_fn, dtype={"source":str, 
                                                "target":str, 
                                                "source_tweet":str, 
                                                "target_tweet":str, 
                                                "tweet1":str, 
                                                "tweet2":str})
        
        # Preprocess text by removing mentions
        self.data["tweet1"] = self.data["tweet1"].apply(self.remove_mentions)
        self.data["tweet2"] = self.data["tweet2"].apply(self.remove_mentions)
        
        # Class 1 - less than or equal to 3 hops
        # Class 0 - greater than 3 hops
        self.data["class"] = (self.data["sp"] < sp_thres).astype(float)
            
        # Randomize the dataframe
        self.data = self.data.sample(frac=1).reset_index(drop=True)
   
    def remove_mentions(self, string):
        '''
        Function to remove mentions from text
        '''
        string = re.sub(r"@[A-Za-z0-9_]+","", string)
        string = string.strip()
        return string
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        user1id = self.data.iloc[idx]["source"]
        user2id = self.data.iloc[idx]["target"]
        tweet1id = self.data.iloc[idx]["source_tweet"]
        tweet2id = self.data.iloc[idx]["target_tweet"]
        tweet1 = self.data.iloc[idx]["tweet1"]
        tweet2 = self.data.iloc[idx]["tweet2"]
        value = torch.tensor([int(self.data.iloc[idx]["sp"])])
        label = torch.tensor([int(self.data.iloc[idx]["class"])])
        
        return user1id, user2id, tweet1id, tweet2id, tweet1, tweet2, value, label  