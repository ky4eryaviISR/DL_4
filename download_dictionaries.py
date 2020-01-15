import os

if not os.path.exists('data/glove'):
    os.system(f'wget http://nlp.stanford.edu/data/glove.6B.zip && '
              f'unzip glove.6B.zip && '
              f'mkdir data/glove && '
              f'mv glove*.txt data/glove')
if not os.path.exists('data/fasttext'):
  os.system(f'wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip &&'
            f'unzip wiki-news-300d-1M.vec.zip && '
            f'mkdir data/fasttext && '
            f'mv data/wiki-news-300d-1M.vec data/fasttext')
if not os.path.exists('data/word2vec'):
  os.system(f'wget https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz && '
            f'gunzip GoogleNews-vectors-negative300.bin.gz && '
            f' mkdir data/word2vec && '
            f'mv GoogleNews-vectors-negative300.bin data/word2vec')
