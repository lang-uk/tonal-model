# -*- coding: utf-8 -*-

"""
builds a vocabulary for word2vec and saves it
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

# create empty model
model = gensim.models.Word2Vec(size=200)

log('building dictionary')
model.build_vocab(all_sentences)

log('saving model')
model.save(trained_folder + 'word2vec_200_dmt.model')

log('done', Fore.GREEN)
