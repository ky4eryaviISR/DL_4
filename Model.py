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


def is_accessible(path):
    """
    Check if the file or directory at `path` can
    be accessed by the program using `mode` open flags.
    """
    try:
        p = Path(path)
        new_name = ''.join([p.stem, '_minimize.npy'])
        input_f = Path(p.parent, f"{new_name}").as_posix()
        my_dict_back = np.load(input_f, allow_pickle=True)
    except IOError:
        return None
    return my_dict_back.item()


def parse_embedding(path):
    glove_index = is_accessible(path)
    if not glove_index:
        glove_index = {}
        n_lines = sum(1 for _ in open(path, encoding='utf8'))
        with open(path, encoding='utf8') as fp:
            for line in tqdm(fp, total=n_lines):
                split = line.split()
                word = split[0]
                vec = np.array(split[1:], dtype='float32')
                vector = vec
                glove_index[word] = vector
    return glove_index


class MetaEmbedding(nn.Module):
    def __init__(self, global_dict):
        super().__init__()
        dim_fasttext = 300

        path = 'data/fasttext/wiki-news-300d-1M.vec'
        print("Start parsing fasttext")
        fasttext = parse_embedding(path)
        fasttext = self.create_embedding(global_dict, fasttext, dim_fasttext, path)

        self.fasttext = nn.Embedding.from_pretrained(fasttext, freeze=False)
        print("Start parsing glove")
        dim_glove = 300
        path = 'data/glove/glove.6B.300d.txt'
        glove = parse_embedding(path)
        glove = self.create_embedding(global_dict, glove, dim_glove, path)
        dim_glove = glove.shape[1]
        self.glove = nn.Embedding.from_pretrained(glove, freeze=False)

        self.proj_fasttext = nn.Linear(dim_fasttext, 300)
        nn.init.xavier_normal_(self.proj_fasttext.weight)
        self.proj_glove = nn.Linear(dim_glove, 300)
        nn.init.xavier_normal_(self.proj_glove.weight)

        self.fasttext_get_alpha = nn.Linear(dim_fasttext, 1)
        self.glove_get_aplha = nn.Linear(dim_glove, 1)

    def create_embedding(self, word_dict, embedding_files, dim, path):
        embed = torch.zeros(len(word_dict.items()), dim)
        glove_index = {}
        statistic = {'exact': 0, 'lower': 0, 'new vector': 0}
        for word, loc in tqdm(word_dict.items()):
            if word in embedding_files.keys():
                statistic['exact'] += 1
                vector = torch.from_numpy(embedding_files[word])
                embed[loc-1, :] = vector[:]
                glove_index[word] = embedding_files[word]
            elif word.lower() in embedding_files.keys():
                statistic['lower'] += 1
                glove_index[word.lower()] = embedding_files[word]
            else:
                statistic['new vector'] += 1
                glove_index[word] = torch.FloatTensor(300).uniform_(-1, 1)
        print('Statistic:', statistic)
        p = Path(path)
        new_name = ''.join([p.stem, '_minimize.npy'])
        outfile = Path(p.parent, f"{new_name}").as_posix()
        np.save(outfile, glove_index)
        return embed

    def forward(self, word):
        glove_out = self.proj_glove(self.glove(word))
        fast_out = self.proj_fasttext(self.fasttext(word))
        glove_alpha = self.glove_get_aplha(glove_out)
        fast_alpha = self.fasttext_get_alpha(fast_out)
        embed = torch.cat([glove_out.unsqueeze(2), fast_out.unsqueeze(2)], dim=2)
        alpha = torch.cat([glove_alpha, fast_alpha], dim=2)
        alpha = F.softmax(alpha, dim=2).unsqueeze(3).expand_as(embed)
        out = alpha*embed
        out = out.sum(dim=2)
        return torch.sigmoid(out)


class SentenceEncoder(nn.Module):
    def __init__(self, embedding_dim, out_dim):
        super().__init__()
        self.bilstm = nn.LSTM(embedding_dim, out_dim, bidirectional=True, num_layers=1)

    def init_hidden(self, batch_size=1):
        return (torch.randn(2, batch_size, 128).to(device),
                torch.randn(2, batch_size, 128).to(device))

    def forward(self, words, sen_len):
        out = words
        out = pack_padded_sequence(out, sen_len, batch_first=True, enforce_sorted=False)
        out, _ = self.bilstm(out, self.init_hidden(words.shape[0]))
        out, _ = pad_packed_sequence(out, batch_first=True)
        return out.max(1)[0]


class MainModel(nn.Module):

    def __init__(self, vocabulary_map, emb_dim=300, out_dim=128):
        super().__init__()
        self.dme = MetaEmbedding(vocabulary_map).to(device)
        self.sen_encoder = SentenceEncoder(emb_dim, out_dim).to(device)
        self.classifier = nn.Sequential(
            nn.Linear(out_dim * 2 * 4, 1024),
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
        sen_1 = sen_1.view(prec[0].shape[1], prec[0].shape[0])
        sen_2 = sen_2.view(hyp[0].shape[1], hyp[0].shape[0])
        sen_1 = self.dme(sen_1)
        sen_2 = self.dme(sen_2)

        sen_1 = self.sen_encoder(sen_1, len_1)
        sen_2 = self.sen_encoder(sen_2, len_2)
        out = torch.cat([sen_1, sen_2, sen_1-sen_2, sen_1*sen_2], dim=1)
        return self.classifier(out)


if __name__ == '__main__':
    # sentence = 'the quick brown fox jumps over the lazy dog'
    # words = sentence.split()
    #
    GLOVE_FILENAME = 'glove/glove.6B.50d.txt'
    # name = Path(GLOVE_FILENAME).stem
    #
    # glove_index = {}
    # n_lines = sum(1 for line in open(GLOVE_FILENAME))
    # with open(GLOVE_FILENAME) as fp:
    #     for line in tqdm(fp, total=n_lines):
    #         split = line.split()
    #         word = split[0]
    #         vector = np.array(split[1:]).astype(float)
    #         glove_index[word] = vector
    #
    words = {'the': 1, 'country': 2, 'box': 3}

    #glove_embeddings = np.array([glove_index[word] for word in words])


    embed = MetaEmbedding(words)