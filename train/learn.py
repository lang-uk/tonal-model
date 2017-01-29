# -*- coding: utf-8 -*-

"""
loads model with vocabulary saved by build_w2v_dict.py
trains a model on sentences and saves it
"""
import sys
sys.path.append('../')

import gensim, logging
from utils import *
from colorama import Fore, Back, Style

log('starting')

# create log directory
log_root_folder = create_log_folder(__file__)

# setup logging to write to current log folder
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                    filename=log_root_folder + 'word2vec.log')

# load model with dictionary from file
model = gensim.models.Word2Vec.load(trained_folder + 'big_word2vec_200.model')

log('training')
model.train(all_sentences)

log('saving model')
model.save(trained_folder + 'big_word2vec_trained_200.model')

log('done', Fore.GREEN)
