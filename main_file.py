import io

from dataloader import SNLI_DataLoader


# def load_vectors(fname):
#     fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
#     n, d = map(int, fin.readline().split())
#     data = {}
#     for line in fin:
#         tokens = line.rstrip().split(' ')
#         data[tokens[0]] = map(float, tokens[1:])
#     return data
#
# load_vectors('data/fasttext300d-2m/crawl-300d-2M.vec')


if __name__=='__main__':
    dt = SNLI_DataLoader()
    print('x')
