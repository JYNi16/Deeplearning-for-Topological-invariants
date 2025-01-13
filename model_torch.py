# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 13:23:31 2025

@author: Curry
"""

import torch 
from torch import nn
import numpy as np

#import torch.optim.lr_scheduler import StepLR



class CNN_TI(nn.Module):
    def __init__(self):
        
        super(CNN_TI, self).__init__()
        self.c1 = nn.Conv2d(1, 40, 2)
        self.r1 = nn.ReLU()
        self.c2 = nn.Conv2d(40, 1, 1)
        self.r2 = nn.ReLU()
        
        self.f1 = nn.Flatten()
        self.d1 = nn.Dropout()
        
        self.l1 = nn.Linear(32, 16)
        self.r3 = nn.ReLU()
        self.l2 = nn.Linear(16, 1)
    
    def forward(self, x):
        
        x = self.c1(x)
        x = self.r1(x)
        
        x = self.c2(x)
        x = self.r2(x)
        
        x = self.f1(x)
        x = self.d1(x)
        x = self.l1(x)
        x = self.r3(x)
        x = self.l2(x)
        
        return x
        


if __name__=="__main__":
    model = CNN_TI()
    a = torch.ones(1, 1, 33, 2)
    out = model(a)
    print("out.shape is:", out.shape)
        
        
