# -*- coding: utf-8 -*-

"""
load chunks one by one and save to chunks with array of sentences each sentences is an array of words
in simple words, the goal this script is a tokanization of text in chunks
"""
import sys
sys.path.append('../')

import pandas as pd
from utils import *
import tokenize_uk
from colorama import Fore, Back, Style

items_processed = 0
chunks_folder = data_folder + 'chunks/'
sents_folder = data_folder + 'sents/'
log_interval = 10000

log('starting')

start_time = time.time()

for chunk_file in os.listdir(chunks_folder):
    data = pd.read_csv(chunks_folder + chunk_file)
    sentences = []

    for row in data.itertuples():
        title = row[3].decode('utf-8') if isinstance(row[3], basestring) else ''
        text = row[2].decode('utf-8') if isinstance(row[2], basestring) else ''

        tokens_title = tokenize_uk.tokenize_sents(title)
        tokens_text = tokenize_uk.tokenize_sents(text)

        for sentence in tokens_title + tokens_text:
            sentences.append(tokenize_uk.tokenize_words(sentence))

        total_time = time.time() - start_time

        if items_processed % log_interval == 0:
            log('items processed {}, running time {}'.format(items_processed, total_time))

        items_processed += 1

    log('saving chunk ')

    result_file = os.path.splitext(chunk_file)[0] + '.msg'
    with open(sents_folder + result_file, 'wb') as f:
        msgpack.pack(sentences, f)

log('done', Fore.GREEN)
