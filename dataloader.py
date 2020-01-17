import torch
from torchtext.data import Field, BucketIterator, Iterator
from torchtext.datasets import SNLI
from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'
print("Graphical device test: {}".format(torch.cuda.is_available()))
print("{} available".format(device))


class SNLI_DataLoader(object):

    def __init__(self):
        self.TEXT = Field(lower=True, include_lengths=True, tokenize=lambda x: x.split())
        self.LABEL = Field(sequential=False, is_target=True, unk_token=None)#, use_vocab=False, is_target=True, unk_token=None, pad_token=None)
        self.test_iter = self.train_iter = self.val_iter = None
        self.load_datasets()

    def load_datasets(self):
        trn, vld, test = SNLI.splits(self.TEXT, self.LABEL, root='data',
                                     train='snli_1.0_train.jsonl',
                                     validation="snli_1.0_dev.jsonl",
                                     test="snli_1.0_test.jsonl")

        self.train_iter, self.val_iter = BucketIterator.splits((trn, vld),
                                                               batch_sizes=(256, 256),
                                                               device=device,
                                                               sort_key=lambda x: (len(x.premise), len(x.hypothesis)),
                                                               sort_within_batch=True,
                                                               repeat=False)

        self.test_iter = Iterator(test, batch_size=64, device=device, sort=False, sort_within_batch=False, repeat=False)
        self.train_iter.shuffle = True
        self.TEXT.build_vocab(trn, vld, test)
        self.LABEL.build_vocab(trn)

    def get_text_2_id_vocabulary(self):
        return self.TEXT.vocab.stoi

    def get_id_2_text_vocabulary(self):
        return self.TEXT.vocab.itos

    def get_label_vocabulary(self):
        return self.LABEL.vocab.stoi
