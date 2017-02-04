# -*- coding: utf-8 -*-

"""
requires trained lexvec with word2vec models placed to ../data/result/ folder
it will join them and save syn0, index2word and joined_dict files
you can find download links for models here: https://github.com/lang-uk/tonal-model/#used-word-embeddings-models
"""

import sys
import os

import numpy as np
from gensim.models import Word2Vec

currentDirectory = os.path.dirname(__file__)
sys.path.append(os.path.join(currentDirectory, '../')) # This will get you to source

from utils import *

log('loading lexVecModel')
lexVecModel = Word2Vec.load_word2vec_format(result_folder + 'lexvec', binary=False)
lexVecModel.init_sims(replace=True)
log('done')

log('loading word2vec')
word2vecModel = Word2Vec.load_word2vec_format(result_folder + 'word2vec', binary=False)
word2vecModel.init_sims(replace=True)
log('done')

# maps word to vector
joined_dict = {}

# 2d array of vectors
syn0 = []

# 1d array of words
index2word = []

log_interval = int(10e3)
processed_words = 0

for word in lexVecModel.vocab:
    if word not in word2vecModel.vocab:
        continue

    joined_vec = np.concatenate((lexVecModel[word], word2vecModel[word])).tolist()
    joined_dict[word] = joined_vec
    index2word.append(word)
    syn0.append(joined_vec)

    processed_words += 1

    if not processed_words % log_interval:
        log('words processed {:5d} out of {:5d}'.format(processed_words, len(lexVecModel.vocab)))

print('saving')

with open(result_folder + 'joined_dict', 'wb') as f:
    msgpack.pack(joined_dict, f)

with open(result_folder + 'index2word', 'wb') as f:
    msgpack.pack(index2word, f)

with open(result_folder + 'syn0', 'wb') as f:
    msgpack.pack(syn0, f)

print('done')
