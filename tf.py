#!/usr/bin/env python
# coding: utf-8

# In[12]:



import matplotlib.pyplot as plt
import cv2
import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

import os

print(tf.version.VERSION)

model = tf.keras.models.load_model('../mr_model.h5')
# model.summary()

# Dictionary for getting characters from index values...
word_dict = {0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z'}
path, dirs, files = next(os.walk("../letters"))
file_count = len(files)
for x in range (file_count):

    img = cv2.imread(f'../letters/letter{x}.png')
    img_copy = img.copy()
    

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (400,440))

    img_copy = cv2.GaussianBlur(img_copy, (7,7), 0)
    img_gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    _, img_thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)

    img_final = cv2.resize(img_thresh, (28,28))
    img_final =np.reshape(img_final, (1,28,28,1))

    img_pred = word_dict[np.argmax(model.predict(img_final))]
    f = open("../sth.txt", "a")
    f.write(img_pred)
    f.close()


# In[ ]:




