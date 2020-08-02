# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 13:36:41 2020

@author: Jesse Havens
"""
import os
import cv2
from tqdm import tqdm as pbar
import numpy as np

IMG_SIZE = 50 #Need uniform images, will make 50x50
CATS = "PetImages/Cat"
DOGS = "PetImages/Dog"
LABELS = {CATS: 0, DOGS: 1}

training_data = []
catcount = 0
dogcount = 0

for label in LABELS:
    for f in pbar(os.listdir(label)):
        try:
            path = os.path.join(label,f)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            training_data.append([np.array(img), np.eye(2)[LABELS[label]]])
            
            if  label == CATS:
                catcount +=1
            elif label == DOGS:
                dogcount +=1
        except Exception as e:
            pass
            #print(str(e)) - can print error but its just bad pics in this case
        
np.random.shuffle(training_data)
np.save("training_data.npy", training_data)
print("Cats: ", catcount)
print("Dogs: ", dogcount)
        
