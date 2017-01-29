# -*- coding: utf-8 -*-

"""
clean data by removing special non alphabetic characters and converting everything to lowercase
"""
import sys
sys.path.append('../')

import re

from utils import *
from colorama import Fore, Back, Style


items_processed = 0
clean_folder = data_folder + 'clean/'
log_interval = 100000

log('starting')

start_time = time.time()

non_char_re = re.compile('\W', re.UNICODE)


def convert_number(num):
    try:
        inum = int(num)

        if inum < 10:
            return str(inum)
        else:
            return "0" * min(len(num), 5)

    except ValueError:
        return num

for file_name in os.listdir(sents_folder):
    with open(sents_folder + file_name, 'rb') as f:
        sentences = msgpack.unpack(f)

    for i, sentence in enumerate(sentences):
        for j, word in enumerate(sentence):
            word = word.decode('utf-8')
            sentence[j] = None if non_char_re.search(word) else convert_number(word.lower())

        sentences[i] = get_not_none(sentence)

        if len(sentences[i]) == 0:
            sentences[i] = None

        total_time = time.time() - start_time

        if items_processed % log_interval == 0:
            log('items processed {} and it took {}'.format(items_processed, total_time))

        items_processed += 1

    sentences = get_not_none(sentences)

    log('saving chunk ' + file_name)

    with open(clean_folder + file_name, 'wb') as f:
        msgpack.pack(sentences, f)

    log('chunk ' + file_name + ' saved', Fore.GREEN)

log('done', Fore.GREEN)
