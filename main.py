import numpy as np
import torch 
from torch import nn
from torch.utils.data import Dataset, DataLoader
from model_torch import CNN_TI
from dataloader import HamData

path_train = "train_data1"
path_val = "val_test1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

val_data = HamData(path_val)
val_data_loader = DataLoader(val_data, batch_size = 64, shuffle = True)

train_data = HamData(path_train)
train_data_loader = DataLoader(train_data, batch_size = 256, shuffle = True)

model = CNN_TI()
loss_function = nn.MSELoss()
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr = 0.005)


def val_test():
    model.eval()
    torch.set_grad_enabled(False)

    loss_val_all = 0
    step = 0

    for x, y in val_data_loader:
        step +=1 
        y_pre = model(x.to(device, torch.float32))
        loss_val_all += loss_function(y_pre,  y.to(device, torch.float32))
    
    return loss_val_all.data.cpu().numpy()/step 


def train_wn():
    loss_all = 0
    step = 0
    
    for x, y in train_data_loader:
        step += 1
        torch.set_grad_enabled(True)
        model.train()
        optimizer.zero_grad()
        y_pre = model(x.to(device, torch.float32))
        loss = loss_function(y_pre,  y.to(device, torch.float32))
        loss.backward()
        optimizer.step()

        loss_all += loss
        
    return loss_all.data.cpu().numpy()/step 


if __name__=="__main__":
    for epoch in range(50):
        loss_train = train_wn()
        print("epoch is {} | train loss is:{}".format(epoch, loss_train))
        loss_val = val_test()
        print("epoch is {} | val loss is:{}".format(epoch, loss_val))


