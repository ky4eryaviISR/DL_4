import os

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

def is_accessible(path):
    """
    Check if the file or directory at `path` can
    be accessed by the program using `mode` open flags.
    """
    try:
        p = Path(path)
        new_name = ''.join([p.stem, '_minimize.npy'])
        input_f = Path(p.parent, f"{new_name}").as_posix()
        my_dict_back = {line.split()[0]: np.array([float(i) for i in line.split()[1:]]) for line in open(input_f)}
    except IOError:
        return None
    return my_dict_back


def parse_embedding(path, vocab):
    glove_index = is_accessible(path)
    if not glove_index:
        glove_index = {}
        n_lines = sum(1 for _ in open(path, encoding='utf8'))
        with open(path, encoding='utf8') as fp:
            for line in tqdm(fp, total=n_lines):
                split = line.split()
                word = split[0]
                if len(split[1:]) != 300 or (word not in vocab and word.lower() not in vocab):
                    continue
                vec = np.array(split[1:], dtype='float32')
                vector = vec
                glove_index[word] = vector
        p = Path(path)
        new_name = ''.join([p.stem, '_minimize.npy'])
        outfile = Path(p.parent, f"{new_name}").as_posix()
        if not os.path.exists(outfile):
            with open(outfile, 'w') as fp:
                fp.writelines('\n'.join([k+' ' + ' '.join([str(i) for i in v])
                                        for k, v in glove_index.items()]))
    return glove_index


class MetaEmbedding(nn.Module):
    def __init__(self, global_dict):
        super().__init__()
        dim_fasttext = 300
        print("Start parsing fasttext")
        fasttext = parse_embedding(EMB_FASTTEXT, global_dict)
        fasttext = self.create_embedding(global_dict, fasttext, dim_fasttext, EMB_FASTTEXT)

        self.fasttext = nn.Embedding.from_pretrained(fasttext, freeze=True)
        print("Start parsing glove")
        dim_glove = 300
        glove = parse_embedding(EMB_GLOVE, global_dict)
        glove = self.create_embedding(global_dict, glove, dim_glove, EMB_GLOVE)
        dim_glove = glove.shape[1]
        self.glove = nn.Embedding.from_pretrained(glove, freeze=True)

        self.proj_fasttext = nn.Linear(dim_fasttext, 300)
        nn.init.xavier_normal_(self.proj_fasttext.weight)
        self.proj_glove = nn.Linear(dim_glove, 300)
        nn.init.xavier_normal_(self.proj_glove.weight)
        self.fasttext_get_alpha = nn.Linear(dim_fasttext, 1)
        self.glove_get_aplha = nn.Linear(dim_glove, 1)

    def create_embedding(self, word_dict, embedding_files, dim, path):
        embed = torch.FloatTensor(len(word_dict.items()), dim).uniform_(-1, 1)
        for word, loc in tqdm(word_dict.items()):
            if word in embedding_files.keys():
                vector = torch.from_numpy(embedding_files[word])
                embed[loc-1, :] = vector[:]
        return embed

    def forward(self, word):
        glove_out = self.proj_glove(self.glove(word))
        fast_out = self.proj_fasttext(self.fasttext(word))
        glove_alpha = self.glove_get_aplha(glove_out)
        fast_alpha = self.fasttext_get_alpha(fast_out)
        embed = torch.stack([glove_out, fast_out], dim=2)
        alpha = torch.cat([glove_alpha, fast_alpha], dim=2)
        alpha = F.softmax(alpha, dim=2).unsqueeze(3).expand_as(embed)
        out = alpha*embed
        out = out.sum(dim=2)
        return out


class SentenceEncoder(nn.Module):
    def __init__(self, embedding_dim, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.bilstm = nn.LSTM(embedding_dim, out_dim, bidirectional=True, num_layers=1)

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
        self.dme = MetaEmbedding(vocabulary_map).to(device)
        self.sen_encoder = SentenceEncoder(emb_dim, out_dim).to(device)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(out_dim*4*2, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 3),
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
