import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from enum import Enum
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm
import numpy as np



class MetaEmbedding(nn.Module):
    def __init__(self, name, embedding_files, global_dict):
        super().__init__()



        choice = {'glove.6B.50d': None}
        word_dict = choice[name]


        self.embedding = MetaEmbedding.create_embedding(global_dict, embedding_files)


    @staticmethod
    def create_embedding(word_dict, embedding_files):
        embed = torch.zeros(len(word_dict.items()), 50)
        for word, loc in word_dict.items():
            if word in embedding_files.keys():
                vector = torch.from_numpy(embedding_files['word'])
                embed[loc-1, :] = vector[:]
        return embed


    def forward(self, word):
        out = []
        for emb in self.meta_emb:
            out.append(emb(word))
        return torch.cat([vec for vec in out])


class SentenceEncoder(nn.Module):
    def __init__(self, embedding_dim, out_dim):
        super().__init__()
        self.bilstm = nn.LSTM(embedding_dim, out_dim, bidirectional=True, num_layers=1, dropout=0.3)
        self.max_pool_layer = nn.MaxPool1d()

    def forward(self, words, sen_len):
        out = words
        out = pack_padded_sequence(out, sen_len, batch_first=True, enforce_sorted=False)
        out, _ = self.bilstm(out)
        out, _ = pad_packed_sequence(out, batch_first=True)
        out = self.max_pool_layer(out)
        return out


class MainModel(nn.Module):

    def __init__(self, embeddings, vocabulary_map, emb_dim=512, out_dim=1024, ):
        super().__init__()
        self.dme = MetaEmbedding(embeddings)
        self.sen_encoder = SentenceEncoder(emb_dim, out_dim)

    def forward(self, sentences, sen_len):
        out = sentences
        out = self.dme(out)
        self.sen_encoder(out)



if __name__ == '__main__':
    sentence = 'the quick brown fox jumps over the lazy dog'
    words = sentence.split()

    GLOVE_FILENAME = 'glove/glove.6B.50d.txt'
    name = Path(GLOVE_FILENAME).stem

    glove_index = {}
    n_lines = sum(1 for line in open(GLOVE_FILENAME))
    with open(GLOVE_FILENAME) as fp:
        for line in tqdm(fp, total=n_lines):
            split = line.split()
            word = split[0]
            vector = np.array(split[1:]).astype(float)
            glove_index[word] = vector

    words = {'the': 1, 'country': 2, 'box': 3}

    #glove_embeddings = np.array([glove_index[word] for word in words])


    embed = MetaEmbedding(name, glove_index, words)