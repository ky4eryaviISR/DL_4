import os
from tqdm import tqdm
import numpy as np

if not os.path.exists('glove'):
    os.system(f'wget http://nlp.stanford.edu/data/glove.6B.zip && '
              f'unzip glove.6B.zip && '
              f'mkdir glove && '
              f'mv glove*.txt')
if not os.path.exists('fasttext'):
  os.system(f'wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip &&'
            f'unzip wiki-news-300d-1M.vec.zip && '
            f'mkdir fasttext && '
            f'mv wiki-news-300d-1M.vec fasttext')
if not os.path.exists('word2vec'):
  os.system(f'wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz && '
            f'gunzip GoogleNews-vectors-negative300.bin.gz && '
            f' mkdir word2vec && '
            f'mv GoogleNews-vectors-negative300.bin word2vec')
