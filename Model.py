import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class MetaEmbedding(nn.Module):
    def __init__(self, embedding_files):
        super().__init__()
        self.meta_emb = []
        for emb in embedding_files:
            self.meta_emb.append(nn.Embedding.from_pretrained(emb))

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

    def __init__(self, embeddings,emb_dim=512, out_dim=1024):
        super().__init__()
        self.dme = MetaEmbedding(embeddings)
        self.sen_encoder = SentenceEncoder(emb_dim, out_dim)

    def forward(self, sentences, sen_len):
        out = sentences
        out = self.dme(out)
        self.sen_encoder(out)

