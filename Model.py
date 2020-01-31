import os
import pickle

import joblib
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.functional import F
from pathlib import Path
from tqdm import tqdm
import numpy as np
from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'

EMB_GLOVE = 'data/glove/glove.840B.300d.txt'
EMB_FASTTEXT = 'data/fasttext/crawl-300d-2M.vec'
# EMB_LEVY = 'data/levy/bow2.words'

def is_accessible(path):
    """
    Check if the file or directory at `path` can
    be accessed by the program using `mode` open flags.
    """
    try:
        p = Path(path)
        new_name = ''.join([p.stem, '_minimize.npy'])
        input_f = Path(p.parent, f"{new_name}").as_posix()
        my_dict_back = joblib.load(open(input_f, 'rb'))
    except IOError:
        return None
    return my_dict_back


def parse_embedding(path):
    glove_index = is_accessible(path)
    if not glove_index:
        glove_index = {}
        n_lines = sum(1 for _ in open(path, encoding='utf8'))
        with open(path, encoding='utf8') as fp:
            for line in tqdm(fp, total=n_lines):
                split = line.split()
                word = split[0]
                if len(split[1:]) != 300:
                    continue
                vec = np.array(split[1:], dtype='float32')
                glove_index[word] = torch.tensor(vec)
    return glove_index


def normalize(embedding, norm, miss):
    if len(norm) > 0:
        embedding[norm, :] = embedding[norm, :] - embedding[norm, :].mean(0)
    if len(miss) > 0:
        embedding[miss] = 0
    return embedding


class MetaEmbedding(nn.Module):

    def __init__(self, global_dict, emb, padding):
        super().__init__()

        dim = 300

        print("Start parsing fasttext")
        fasttext = parse_embedding(EMB_FASTTEXT)
        fasttext, norm, miss = self.create_embedding(global_dict, fasttext, dim, EMB_FASTTEXT)
        fasttext = normalize(fasttext, norm, miss)
        self.fasttext = nn.Embedding.from_pretrained(fasttext, freeze=True, padding_idx=padding)

        print("Start parsing glove")
        glove = parse_embedding(EMB_GLOVE)
        glove, norm, miss = self.create_embedding(global_dict, glove, dim, EMB_GLOVE)
        glove = normalize(fasttext, norm, miss)
        self.glove = nn.Embedding.from_pretrained(glove, freeze=True, padding_idx=padding)

        print("Start parsing levy")
        # levy = parse_embedding(EMB_LEVY, global_dict)
        # levy, norm, miss = self.create_embedding(global_dict, levy, dim)
        # levy = normalize(fasttext, norm, miss)
        # self.levy = nn.Embedding.from_pretrained(levy, freeze=True, padding_idx=padding)

        self.proj_fasttext = nn.Linear(dim, emb)
        nn.init.xavier_normal_(self.proj_fasttext.weight)
        self.proj_glove = nn.Linear(dim, emb)
        nn.init.xavier_normal_(self.proj_glove.weight)

        self.fasttext_get_alpha = nn.Sequential(nn.Linear(dim, 10),
                                                nn.Linear(10, 1))
        self.glove_get_aplha = nn.Sequential(nn.Linear(dim, 10),
                                             nn.Linear(10, 1))

    def create_embedding(self, word_dict, embedding_files, dim, path):
        to_norm = []
        miss = []
        emb_store = {}
        embed = torch.FloatTensor(len(word_dict.items()), dim).zero_()
        for word, loc in tqdm(word_dict.items()):
            if word in embedding_files.keys():
                embed[loc-1, :] = embedding_files[word]
                to_norm.append(loc-1)
                emb_store[word] = embedding_files[word]

            elif word.lower() in embedding_files.keys():
                embed[loc - 1, :] = embedding_files[word.lower()]
                to_norm.append(loc - 1)
                emb_store[word.lower()] = embedding_files[word.lower()]

            else:
                print(word)
                miss.append(loc-1)
        p = Path(path)
        new_name = ''.join([p.stem, '_minimize.npy'])
        outfile = Path(p.parent, f"{new_name}").as_posix()
        if not os.path.exists(outfile):
            pickle.dump(emb_store, open(outfile, 'wb'))
        return embed, to_norm, miss

    def forward(self, word):
        glove_emb = self.glove(word)
        glove_out = self.proj_glove(glove_emb)
        fast_emb = self.fasttext(word)
        fast_out = self.proj_fasttext(fast_emb)

        embed = torch.stack([glove_out, fast_out], dim=2)
        alpha = self.fasttext_get_alpha(embed)
        alpha = F.softmax(alpha, dim=2).expand_as(embed)
        out = alpha*embed
        out = out.sum(dim=2)
        return F.relu(out)


class SentenceEncoder(nn.Module):
    def __init__(self, embedding_dim, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.bilstm = nn.LSTM(embedding_dim, out_dim, bidirectional=True, num_layers=1)
        nn.init.orthogonal_(self.bilstm.weight_hh_l0)
        nn.init.orthogonal_(self.bilstm.weight_ih_l0)

    def init_hidden(self, batch_size=1):
        return (torch.randn(2, batch_size, self.out_dim).to(device),
                torch.randn(2, batch_size, self.out_dim).to(device))

    def forward(self, words, sen_len):
        out = words
        sen_len, idx_sort = torch.sort(sen_len, descending=True)
        _, idx_unsort = torch.sort(idx_sort, descending=False)
        out = out.index_select(1, torch.autograd.Variable(idx_sort))
        out = pack_padded_sequence(out, sen_len)
        out, _ = self.bilstm(out, self.init_hidden(words.shape[1]))
        out, _ = pad_packed_sequence(out)
        out = out.index_select(1, torch.autograd.Variable(idx_unsort))

        return out.max(0)[0]


class MainModel(nn.Module):

    def __init__(self, vocabulary_map, emb_dim=300, out_dim=256):
        super().__init__()
        self.dme = MetaEmbedding(vocabulary_map, emb_dim, vocabulary_map['<pad>']).to(device)
        self.sen_encoder = SentenceEncoder(emb_dim, out_dim).to(device)
        #MLP
        self.classifier = nn.Sequential(
            nn.Linear(out_dim * 4 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3),
        )

    def forward(self, prec, hyp):
        sen_1, len_1 = prec
        sen_2, len_2 = hyp
        sen_1 = self.dme(sen_1)
        sen_2 = self.dme(sen_2)

        sen_1 = self.sen_encoder(sen_1, len_1)
        sen_2 = self.sen_encoder(sen_2, len_2)
        out = torch.cat([sen_1, sen_2, sen_1-sen_2, sen_1*sen_2], dim=1)
        return self.classifier(out)