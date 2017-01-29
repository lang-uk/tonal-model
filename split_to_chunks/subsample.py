# -*- coding: cp1251 -*-

import sys
sys.path.append('../')
from utils import *
from colorama import Fore, Back, Style

chunk_size = 5000

files = [
         '../raw src data/wiki_sent_tok.txt',
         # '../raw src data/up.csv',
         # '../raw src data/zaxid.csv',
         # '../raw src data/dt.all.txt',
         # '../raw src data/um.txt',
         # '../raw src data/vz.txt',
         # '../raw src data/korr.csv',
         # '../raw src data/unian.csv',
    ]

log('starting')

for src_file in files:
    log('reading src file ' + src_file)

    content = ''
    lines_read = 0
    with open(src_file, 'r') as f:
        for line in f:
            content += line
            lines_read += 1

            if lines_read == chunk_size:
                break

    log('writing sample to file')
    with open(subsets_folder + os.path.basename(src_file), 'w') as f:
        f.write(content)

    log('writing done', Fore.GREEN)

log('done', Fore.GREEN)
