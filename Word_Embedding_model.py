#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  jan  7 16:29:22 2020

@author: Anvesh
"""

from tensorflow.keras.preprocessing.text import one_hot

sent=['the apple is red in color',
      'glass of milk',
      'the story is good and amusing',
      'we are good programmers',
      'you are smart',]

voc_size= 10000

''''' one _hot representation'''''

one_hot_repr=[one hot(words,voc_size) for words in sent]

'''' word embedding respresentation''''

from tensorflow.keras.layers import embedding

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import sequential

import numpy as np
sent_length=8

embedded_docs= pad_sequences(onehot_repr,padding='pre',maxlent=sent_length)

dim= 15

model= sequential()
model.add(embedding(voc_size,10,input_length=sent_length))
model.compile('adam','mse')

model.summary()

print(model.predict(embedded_docs))

print(model.predict(embedded_docs)[0])