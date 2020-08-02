# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 10:17:03 2020

@author: jesse
"""

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('model.log',sep=',')
print(data)

plt.figure()
plt.plot(data[['MODEL_TIME']], data[['IN_SAMPLE_ACC']], 'k-', data[['MODEL_TIME']], data[['OUT_SAMPLE_ACC']], 'r-')
plt.xlabel('Time')
plt.ylabel('Accuracy')

plt.figure()
plt.plot(data[['MODEL_TIME']], data[['IN_SAMPLE_LOSS']], 'k-', data[['MODEL_TIME']], data[['OUT_SAMPLE_LOSS']], 'r-')
plt.xlabel('Time')
plt.ylabel('Loss')

plt.show()








