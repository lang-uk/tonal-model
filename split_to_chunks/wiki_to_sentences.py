# -*- coding: utf-8 -*-

"""
in wiki file each line is a sentence, so we can read and tokanize line by line
saves lists of sentences in chunks
"""
import sys
sys.path.append('../')
from itertools import islice

from utils import *
import tokenize_uk
from colorama import Fore, Back, Style

items_processed = 0
files_saved = 0
lines_per_chunk = int(7e5)  # 700 000
log_interval = 10000

files = [raw_data_folder + 'wiki_sent_tok.txt']

log('starting')

for src_file in files:
    log('processing file ' + src_file)

    with open(src_file, 'rb') as f:
        while True:
            next_n_lines = list(islice(f, lines_per_chunk))
            if not next_n_lines:
                break

            text = ''.join(next_n_lines).decode('utf-8')

            tokens_text = tokenize_uk.tokenize_sents(text)

            log('tokenization finished')

            sentences = []

            for sentence in tokens_text:
                sentences.append(tokenize_uk.tokenize_words(sentence))

                if items_processed % log_interval == 0:
                    log('items processed {}'.format(items_processed))

                items_processed += 1

            result_file = os.path.basename(src_file) + str(files_saved) + '.msg'
            with open(sents_folder + result_file, 'wb') as s_f:
                msgpack.pack(sentences, s_f)

            log('file {} saved'.format(result_file))

            files_saved += 1

log('done', Fore.GREEN)
