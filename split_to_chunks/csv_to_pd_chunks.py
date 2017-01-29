# -*- coding: cp1251 -*-
"""
split big files to smaller chunks
each chunk is csv
"""
import sys
sys.path.append('../')
import pandas as pd
import math
from utils import *
from colorama import Fore, Back, Style

chunk_size = int(1e5)  # 100 000
chunks_saved = 0
files = [{
    'url': raw_data_folder + 'ukrlib.csv',
    'cols': [1, 0]
}, {
    'url': raw_data_folder + 'up.csv',
    'cols': [6, 4]
}, {
    'url': raw_data_folder + 'zaxid.csv',
    'cols': [5, 2]
}, {
    'url': raw_data_folder + 'korr.csv',
    'cols': [1, 2]
}, {
    'url': raw_data_folder + 'unian.csv',
    'cols': [5, 4]
}]

log('starting')

for src_file in files:
    log('reading src file ' + src_file['url'])
    data = pd.read_csv(src_file['url'])
    chunks_number = int(math.ceil(data.shape[0] / float(chunk_size)))

    log('chunks to save ' + str(chunks_number))

    for i in range(0, chunks_number):
        start_time = time.time()
        chunk = data.iloc[i * chunk_size: (i + 1) * chunk_size, src_file['cols']]

        log('saving chunk ' + str(i))

        chunk.to_csv(data_folder + 'chunks/chunk_{}.csv'.format(chunks_saved))
        total_time = time.time() - start_time

        log('saving completed {}, running time {}'.format(chunks_saved, total_time), Fore.GREEN)

        chunks_saved += 1

log('done', Fore.GREEN)
