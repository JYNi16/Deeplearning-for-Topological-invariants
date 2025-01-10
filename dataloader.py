import numpy as np
#import tensorflow as tf
#from tensorflow import keras
#from keras import layers, Model, losses
import glob, os
import torch 
import torchvision 
from torch.utils.data import Dataset, DataLoader

class HamData(Dataset):
    def __init__(self, path):
        self.list = []
        self.list.extend(glob.glob(os.path.join(path, "*.npz")))
        self.batch_size = 50
    
    
    def __getitem__(self, index):
        
        print("data path is:", self.list)
        data_path = self.list[index]
        data = np.load(data_path)
        
        Ham = data["s"]
        wn = data["label"]
        
        if wn == "0":
            label = 0
        elif wn == "1":
            label = 1
        else:
            label = 2
            
        
        return Ham, wn 
    
    def __len__(self):
        return len(self.list)
        
        
    

if __name__=="__main__":
    path = "./train_data1"
    train_data = HamData(path)
    
    data = np.load(path + "/train.0.npz")
    
    print("data is:", data["s"])
    
    #train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    
    #for s, label in train_loader:
    #    print("i.shape is:", s.shape)

