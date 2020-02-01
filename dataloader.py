import json
import os

import spacy
import torch
from torchtext.data import Field, BucketIterator, Iterator
from torchtext.datasets import SNLI
from torch import cuda
from tqdm import tqdm

device = 'cuda' if cuda.is_available() else 'cpu'
print("Graphical device test: {}".format(torch.cuda.is_available()))
print("{} available".format(device))


class SNLI_DataLoader(object):
    embedding = None

    def __init__(self, emb_set):
        SNLI_DataLoader.embedding = emb_set
        self.TEXT = Field(include_lengths=True, preprocessing=SNLI_DataLoader.get_emb_index)
        self.LABEL = Field(sequential=False, is_target=True, unk_token=None)#, use_vocab=False, is_target=True, unk_token=None, pad_token=None)
        self.test_iter = self.train_iter = self.val_iter = None
        self.tokenize()
        self.load_datasets()

    @staticmethod
    def get_emb_index(x):
        res = []
        for i in x:
            if i in SNLI_DataLoader.embedding:
                res.append(i)
            elif i.lower() in SNLI_DataLoader.embedding:
                res.append(i.lower())
            else:
                res.append('<unk>')
        return res

    def tokenize(self):
        root_path = os.path.join('data')
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        dir_name = os.path.join('data', 'snli', 'snli_1.0')
        if not os.path.exists(dir_name):
            SNLI.download(root_path)
        file_names = ['snli_1.0_dev',
                      'snli_1.0_test',
                      'snli_1.0_train'
                      ]
        spacy_nlp = spacy.load('en')

        for file_name in file_names:
            p = os.path.join(dir_name, file_name)
            print(p)
            if os.path.exists(p + '.tokenized.jsonl'):
                continue
            with open(p + '.jsonl', mode='r') as f_r, open(p + '.tokenized.jsonl', mode='w') as f_w:
                progress = tqdm(f_r, mininterval=1, leave=False)
                for line in progress:
                    line = json.loads(line)
                    sentence1, sentence2 = line['sentence1'], line['sentence2']
                    sentence1_spacy = ' '.join([t.text for t in spacy_nlp(sentence1)])
                    sentence2_spacy = ' '.join([t.text for t in spacy_nlp(sentence2)])
                    f_w.write(str({'sentence1': sentence1_spacy,
                                   'sentence2': sentence2_spacy,
                                   'gold_label': line['gold_label']}).replace('\'', '"'))

    def load_datasets(self):
        trn, vld, test = SNLI.splits(self.TEXT, self.LABEL, root='data',
                                     train='snli_1.0_train.tokenized.jsonl',
                                     validation="snli_1.0_dev.tokenized.jsonl",
                                     test="snli_1.0_test.tokenized.jsonl")

        self.train_iter, self.val_iter = BucketIterator.splits((trn, vld),
                                                               batch_sizes=(64, 512),
                                                               device=device,
                                                               sort=False,
                                                               repeat=False)

        self.test_iter = Iterator(test, batch_size=512, device=device, sort=False, repeat=False)
        self.train_iter.shuffle = True
        self.TEXT.build_vocab(trn, vld, test, min_freq=1)
        self.LABEL.build_vocab(trn, min_freq=1)

    def get_text_2_id_vocabulary(self):
        return self.TEXT.vocab.stoi

    def get_id_2_text_vocabulary(self):
        return self.TEXT.vocab.itos

    def get_label_vocabulary(self):
        return self.LABEL.vocab.stoi
