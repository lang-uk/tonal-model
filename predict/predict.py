# -*- coding: utf-8 -*-

"""
load syn0, index2word and joined_dict models, train model and predict for all vocabulary
"""

import sys
import os
currentDirectory = os.path.dirname(__file__)
sys.path.append(os.path.join(currentDirectory, '../')) # This will get you to source

from utils import *
from keras.layers import Dense, Dropout
from keras.models import Sequential
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
import codecs
from sklearn.utils import shuffle
import msgpack

log('loading files')

with open(result_folder + 'new_vec/joined_dict', 'rb') as f:
    joined_dict = msgpack.unpack(f, encoding='utf-8')

log('files loaded')

X = []
y = []

log('read original tsv file')
tsv = pd.read_csv(data_folder + 'orig-tone-dict-v2.tsv', sep='\t', header=None, 
                  names=['word', 'tone', '_', '_1'], encoding='utf-8')

log('read and concat appendix tsv file')
appendix = pd.read_csv(result_folder + 'appendix1.tsv', sep='\t', header=None, 
                       names=['word', 'tone'], encoding='utf-8')

tsv = pd.concat([tsv, appendix], axis=0, ignore_index=True)

tsv = tsv.drop_duplicates(subset='word', keep='last')

for index, row in tsv.iterrows():
    w = row['word']
    t = row['tone']
    if w in joined_dict:
        X.append(joined_dict[w])
        y.append(t)

del joined_dict

X = np.array(X)
y = np.array(y, dtype=np.float)

X, y = shuffle(X, y, random_state=0)

np.save(result_folder + 'new_vec/predict/tonedataX.npy', X)
np.save(result_folder + 'new_vec/predict/tonedataY.npy', y)

log('training data is ready')

X = np.load(result_folder + 'new_vec/predict/tonedataX.npy')
y = np.load(result_folder + 'new_vec/predict/tonedataY.npy').clip(-2, 2)
y /= 4.0  # y /= 3.0
y += 0.5
# y = y.clip(0, 1)

model = Sequential()
model.add(Dense(800, activation='relu', input_shape=(600,)))
model.add(Dropout(0.5))
model.add(Dense(300, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='mse')

#model.load_weights('tonePredictorUkr.h5')
log('starting learning')

model.fit(X, y,
          batch_size=1,
          nb_epoch=100,
          verbose=1,
          validation_split=0.03,
          callbacks=[ModelCheckpoint(result_folder + 'new_vec/predict/tonePredictorUkr.h5',
                                     save_best_only=True, monitor='val_loss')])

with open(result_folder + 'new_vec/syn0', 'rb') as f:
    syn0 = np.array(msgpack.unpack(f))

log('predicting')
preds = model.predict(syn0, verbose=1)

log('saving results')
np.save(result_folder + 'new_vec/predict/preds-all.npy', preds)
np.save(result_folder + 'new_vec/predict/preds100000.npy', preds[:100000])

with open(result_folder + 'new_vec/index2word', 'rb') as f:
    index2word = np.array(msgpack.unpack(f, encoding='utf-8'))

preds = np.load(result_folder + 'new_vec/predict/preds100000.npy')
words = index2word[:100000]

dic = []
for i in range(0, len(preds)):
    dic.append([words[i], preds[i][0]])

dic = sorted(dic, key=lambda l: l[1], reverse=True)

with codecs.open(result_folder + 'new_vec/predict/ToneResults100000.txt', "w", "utf-8") as stream:
    for i in range(0, len(dic)):
        stream.write(dic[i][0] + '\t' + str(dic[i][1]) + u"\n")

print('done')
