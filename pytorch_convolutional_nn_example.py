# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 10:20:37 2020

@author: Jesse Havens
"""

import os
import cv2
import numpy as np
from tqdm import tqdm as pbar
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

def main():
    
    MODEL_NAME = f"model-{int(time.time())}" #Add timestamp to ensure uniqueness
    LEARN_RATE = 0.001 #Initial step rate of neural net
    DECAY_RATE = 0.0 #Decay for step rate of neural net
    VALIDATION_PERCENT = 0.1 #Percent of data to separate for validation
    BATCH_SIZE = 100 #Neural net will train on batches of the given size
    EPOCHS = 2 #Full passes through data
    
    #Load training data obtained from 'prepare_dogvscat_data.py'
    training_data = np.load("training_data.npy", allow_pickle=True)

    #Initialize neural net
    net = Net()
    optimizer = optim.Adam(net.parameters(), lr=LEARN_RATE, weight_decay=DECAY_RATE)
    loss_function = nn.MSELoss() #Mean squared error loss function
    
    #Convert data to torch tensors
    X = torch.Tensor([pic[0] for pic in training_data]).view(-1, 50, 50) #IMAGE SIZE = 50x50
    X = X/255.0 #Normalize image colors to 1 (max is 255)
    y = torch.Tensor([answer[1] for answer in training_data])
    
    #Split training and test data
    val_size = int(len(X)*VALIDATION_PERCENT)
    
    train_X = X[:-val_size]
    train_y = y[:-val_size]
    
    test_X = X[-val_size:]
    test_y = y[-val_size:]
    
    #Train and test neural net
    optimizer, net = train_net(MODEL_NAME, BATCH_SIZE, EPOCHS, train_X, train_y, test_X, test_y, optimizer, net, loss_function)

    
def train_net(MODEL_NAME, BATCH_SIZE, EPOCHS, train_datax, train_datay, test_datax, test_datay, optimizer, net, loss_function):
    with open("model.log","a") as f:
        f.write(f"MODEL_NAME,MODEL_TIME,EPOCH,IN_SAMPLE_ACC,IN_SAMPLE_LOSS,OUT_SAMPLE_ACC,OUT_SAMPLE_LOSS\n")
        for epoch in range(EPOCHS):
            to_do = range(0, len(train_datax), BATCH_SIZE) #Loop logic
            for batch in pbar(to_do): #Loop with progress bar
                batch_X = train_datax[batch:batch+BATCH_SIZE].view(-1, 1, 50, 50)
                batch_y = train_datay[batch:batch+BATCH_SIZE]
                
                acc, loss, net, optimizer = fwd_pass(batch_X, batch_y, net, loss_function, optimizer, train=True)
                
                if batch % 50 == 0:    
                    val_acc, val_loss = test_model(test_datax, test_datay, net, loss_function, optimizer, size=100)
                    f.write(f"{MODEL_NAME},{round(time.time(),3)},{epoch},{round(float(acc),2)},{round(float(loss),4)},{round(float(val_acc),2)},{round(float(val_loss),4)}\n")
        return optimizer, net

    
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        #test run through to figure out sizing        
        x = torch.randn(50, 50).view(-1, 1, 50, 50)
        self._to_linear = None
        self.convs(x)
        
        self.fc1 = nn.Linear(self._to_linear, 512)
        self.fc2 = nn.Linear(512, 2)
    
    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2,2))
        
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x
    
    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return F.softmax(x, dim=1) #activation function
    
    
def fwd_pass(X, y, net, loss_function, optimizer, train=False):
    if train:
        net.zero_grad()
    outputs = net(X)
    matches = [torch.argmax(output) == torch.argmax(answer) for output, answer in zip(outputs,y)]
    acc = matches.count(True)/len(matches) #Percentage accuracy
    loss = loss_function(outputs, y)
    if train:
        loss.backward()
        optimizer.step()
        
    return acc, loss, net, optimizer
    
def test_model(test_X, test_y, net, loss_function, optimizer, size=32):
    
    random_start = np.random.randint(len(test_X)-size)
    X, y = test_X[random_start:random_start+size], test_y[random_start:random_start+size]
    with torch.no_grad():
        val_acc, val_loss, _, _ = fwd_pass(X.view(-1,1,50,50), y, net, loss_function, optimizer)
        
    return val_acc, val_loss
    
if __name__=='__main__':
    main()





