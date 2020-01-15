import os
from tqdm import tqdm
import numpy as np


if not os.path.exists('glove'):
  os.system(f'wget http://nlp.stanford.edu/data/glove.6B.zip && '
            f'unzip glove.6B.zip && '
            f'mkdir glove && '
            f'mv glove*.txt')
# if not os.path.exists(root+'fasttext'):
#   !wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip && unzip wiki-news-300d-1M.vec.zip && unzip '/content/'wiki-news-300d-1M.vec.zip && mkdir '/content/drive/My Drive/DL_4/'fasttext && mv '/content/'wiki-news-300d-1M.vec '/content/drive/My Drive/DL_4/'fasttext
# if not os.path.exists(root+'word2vec'):
#   !wget "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz" && gunzip /content/GoogleNews-vectors-negative300.bin.gz &&  mkdir '/content/drive/My Drive/DL_4/'word2vec && mv /content/GoogleNews-vectors-negative300.bin '/content/drive/My Drive/DL_4/'word2vec