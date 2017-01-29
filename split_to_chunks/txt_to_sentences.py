# -*- coding: utf-8 -*-

"""
load entire txt file, tokenize it and save lists of sentences in chunks
"""
import sys
sys.path.append('../')

import math

from utils import *
import tokenize_uk
from colorama import Fore, Back, Style

items_processed = 0
chunks_folder = data_folder + 'chunks/'
sents_folder = data_folder + 'sents/'
sents_per_chunk = int(7e5)  # 700 000
log_interval = 10000

files = [raw_data_folder + 'ukr_lit.txt', raw_data_folder + 'td.txt']

log('starting')

for src_file in files:
    with open(src_file, 'rb') as f:
        data = f.read()

    log('processing file ' + src_file)

    text = data.decode('utf-8')
    tokens_text = tokenize_uk.tokenize_sents(text)

    log('tokenization finished')

    sents_number = int(math.ceil(len(tokens_text) / float(sents_per_chunk)))

    for i in range(0, sents_number):
        sentences = []
        chunk = tokens_text[i * sents_per_chunk: (i + 1) * sents_per_chunk]

        for sentence in chunk:
            sentences.append(tokenize_uk.tokenize_words(sentence))

            if items_processed % log_interval == 0:
                log('items processed {}'.format(items_processed))

            items_processed += 1

        result_file = os.path.basename(src_file) + str(i) + '.msg'
        with open(sents_folder + result_file, 'wb') as f:
            msgpack.pack(sentences, f)

        log('file {} saved'.format(result_file))

log('done', Fore.GREEN)
