import torch
from torchtext.data import Field, BucketIterator, Iterator
from torchtext.datasets import SNLI
from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'
print("Graphical device test: {}".format(torch.cuda.is_available()))
print("{} available".format(device))


class SNLI_DataLoader(object):

    def __init__(self):
        self.TEXT = Field(lower=True, tokenize=lambda x: x.split())
        self.LABEL = Field(sequential=True, use_vocab=True)
        self.test_iter = self.train_iter = self.val_iter = None
        self.load_datasets()

    def load_datasets(self):
        trn, vld, test = SNLI.splits(self.TEXT, self.LABEL,root='data',
                                     train='snli_1.0_train.jsonl',
                                     validation="snli_1.0_dev.jsonl",
                                     test="snli_1.0_test.jsonl")

        self.train_iter, self.val_iter = BucketIterator.splits((trn, vld),
                                                               batch_sizes=(64, 64),
                                                               device=device,
                                                               sort_key=lambda x: len(x.premise),
                                                               sort_within_batch=False,
                                                               repeat=False)

        self.test_iter = Iterator(test, batch_size=64, device=device, sort=False, sort_within_batch=False, repeat=False)
        self.TEXT.build_vocab(trn)
        self.LABEL.build_vocab(trn)

    def get_text_2_id_vocabulary(self):
        return self.TEXT.vocab.stoi

    def get_id_2_text_vocabulary(self):
        return self.TEXT.vocab.itos

    def get_label_vocabulary(self):
        return self.LABEL.vocab.stoi
