import sys
import os
sys.path.append('..')

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import codecs
from sklearn.utils import shuffle
import msgpack
from utils import *

log('loading index2word file')
with open(result_folder + 'new_vec/index2word', 'rb') as f:
    index2word = np.array(msgpack.unpack(f, encoding='utf-8'))

log('read original tsv file')
tsv = pd.read_csv(data_folder + 'orig-tone-dict-v2.tsv', sep='\t', header=None, names=['word', 'tone', '_', '_1'])
tones = dict(zip(tsv.word, tsv.tone))

log('loading predictions')
preds = np.load(result_folder + 'new_vec/predict/preds-all.npy')
dic = []
for i in range(0, len(preds)):
    w = index2word[i]
    if w not in tones:
        dic.append([index2word[i], preds[i][0]])

log('sorting')
dic = np.array(sorted(dic, key=lambda l: l[1], reverse=True))

# top positives and negatives
pos = dic[:2000,  :]
neg = dic[-2000:, :][::-1]

with codecs.open(result_folder + 'new_vec/predict/top_pos_N.txt', "w", "utf-8") as stream:
    for i in range(0, len(pos)):
        stream.write(pos[i][0] + '\t' + str(pos[i][1]) + u"\n")

with codecs.open(result_folder + 'new_vec/predict/top_neg_N.txt', "w", "utf-8") as stream:
    for i in range(0, len(neg)):
        stream.write(neg[i][0] + '\t' + str(neg[i][1]) + u"\n")

print('done')
