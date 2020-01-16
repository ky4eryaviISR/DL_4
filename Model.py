import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.functional import F
from pathlib import Path
from tqdm import tqdm
import numpy as np
from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'

def parse_embedding(path):
    glove_index = {}
    n_lines = sum(1 for _ in open(path, encoding='utf8'))
    with open(path, encoding='utf8') as fp:
        for line in tqdm(fp, total=n_lines):
            split = line.split()
            word = split[0]
            vector = np.array(split[1:]).astype(float)
            glove_index[word] = vector
    return glove_index


class MetaEmbedding(nn.Module):
    def __init__(self, embedding_files, global_dict):
        super().__init__()
        dim_fasttext = 300
        fasttext = parse_embedding('data/fasttext/wiki-news-300d-1M.vec')
        fasttext = self.create_embedding(global_dict, fasttext,dim_fasttext)

        self.fasttext = nn.Embedding.from_pretrained(fasttext, freeze=True)
        dim_glove = 50
        glove = parse_embedding('data/glove/glove.6B.50d.txt')
        glove = self.create_embedding(global_dict, glove,dim_glove)
        dim_glove = glove.shape[1]
        self.glove = nn.Embedding.from_pretrained(glove, freeze=True)

        self.proj_fasttext = nn.Linear(dim_fasttext, 256)
        self.proj_glove = nn.Linear(dim_glove, 256)

        self.fasttext_scalar = nn.Parameter(torch.rand(1), requires_grad=True)
        self.glove_scalar = nn.Parameter(torch.rand(1), requires_grad=True)


    def create_embedding(self, word_dict, embedding_files,dim):
        embed = torch.zeros(len(word_dict.items()), dim)
        for word, loc in word_dict.items():
            if word in embedding_files.keys():
                vector = torch.from_numpy(embedding_files['word'])
                embed[loc-1, :] = vector[:]
        return embed

    def forward(self, word):
        glove_out = self.glove_scalar*self.proj_glove(self.glove(word))
        fast_out = self.fasttext_scalar*self.proj_fasttext(self.fasttext(word))
        out = glove_out + fast_out
        return F.sigmoid(out)


class SentenceEncoder(nn.Module):
    def __init__(self, embedding_dim, out_dim):
        super().__init__()
        self.bilstm = nn.LSTM(embedding_dim, out_dim, bidirectional=True, num_layers=1)

    def init_hidden(self, batch_size=1):
        return (torch.randn(2, batch_size, 512).to(device),
                torch.randn(2, batch_size, 512).to(device))

    def forward(self, words, sen_len):
        out = words.view(words.shape[0], words.shape[1], -1)
        out = pack_padded_sequence(out, sen_len, batch_first=True, enforce_sorted=False)
        out, _ = self.bilstm(out, self.init_hidden(words.shape[0]))
        out, _ = pad_packed_sequence(out, batch_first=True)
        return out.max(0)[0]


class MainModel(nn.Module):

    def __init__(self, embeddings, vocabulary_map, emb_dim=256, out_dim=512, ):
        super().__init__()
        self.dme = MetaEmbedding(embeddings, vocabulary_map).to(device)
        self.sen_encoder = SentenceEncoder(emb_dim, out_dim).to(device)
        self.classifier = nn.Sequential(
            nn.Linear(out_dim * 4 * 2, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 3),
        )

    def forward(self, prec, hyp):
        sen_1, len_1 = prec
        sen_2, len_2 = hyp
        sen_1 = sen_1.view(prec[0].shape[1], prec[0].shape[0], -1)
        sen_2 = sen_2.view(hyp[0].shape[1], hyp[0].shape[0], -1)
        sen_1 = self.dme(sen_1)
        sen_2 = self.dme(sen_2)

        sen_1 = self.sen_encoder(sen_1, len_1)
        sen_2 = self.sen_encoder(sen_2, len_2)
        out = torch.cat([sen_1, sen_2, sen_1-sen_2, sen_1*sen_2], dim=1)
        return self.classifier(out)



# if __name__ == '__main__':
#     sentence = 'the quick brown fox jumps over the lazy dog'
#     words = sentence.split()
#
#     GLOVE_FILENAME = 'glove/glove.6B.50d.txt'
#     name = Path(GLOVE_FILENAME).stem
#
#     glove_index = {}
#     n_lines = sum(1 for line in open(GLOVE_FILENAME))
#     with open(GLOVE_FILENAME) as fp:
#         for line in tqdm(fp, total=n_lines):
#             split = line.split()
#             word = split[0]
#             vector = np.array(split[1:]).astype(float)
#             glove_index[word] = vector
#
#     words = {'the': 1, 'country': 2, 'box': 3}
#
#     #glove_embeddings = np.array([glove_index[word] for word in words])


    # embed = MetaEmbedding(name, glove_index, words)