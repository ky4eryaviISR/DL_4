import os

if not os.path.exists('data/glove'):
    os.system(f'wget http://nlp.stanford.edu/data/glove.6B.zip && '
              f'unzip glove.6B.zip && '
              f'mkdir data/glove && '
              f'mv glove.6B.300d.txt data/glove && '
              f'rm -rf glove*.txt && rm -rf glove.6B.zip')

if not os.path.exists('data/fasttext'):
  os.system(f'wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip &&'
            f'unzip crawl-300d-2M.vec.zip && '
            f'mkdir data/fasttext && '
            f'mv crawl-300d-2M.vec data/fasttext && '
            f'rm -rf crawl-300d-2M.vec.zip')

if not os.path.exists('data/levy'):
    os.system(f'wget http://u.cs.biu.ac.il/~yogo/data/syntemb/bow2.words.bz2 && '
              f'bzip2 -d bow2.words.bz2 && '
              f'mkdir data/levy && '
              f'mv bow2.words data/levy')
