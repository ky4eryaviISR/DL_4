import os
import pickle

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm
from Model import MainModel, EMB_FASTTEXT, EMB_GLOVE
from dataloader import SNLI_DataLoader
from torch import cuda, nn
import matplotlib.pyplot as plt
LR = 0.0007
device = 'cuda' if cuda.is_available() else 'cpu'


def save_graph(train, test, y_axis, title=None):
    """
    saving the graph for accuracy/loss
    :param train: list
    :param test: list
    :param y_axis: accuracy/ loss
    :param title: title of the graph
    :return:
    """
    plt.suptitle(y_axis, fontsize=20)
    plt.figure()
    plt.plot(train, color='r', label='train')
    plt.plot(test, color='g', label='validation')
    plt.xlabel('Epochs')
    plt.legend(loc="upper left")
    plt.ylabel(y_axis)
    plt.title(y_axis if not title else title)
    plt.savefig(y_axis+'.png')


def evaluate(model, data_loder, criterion, set_name):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in data_loder:
            output = model(data.premise, data.hypothesis)
            test_loss += criterion(output, data.label).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(data.label.view_as(pred)).sum()
            total += pred.shape[0]
    test_loss /= total
    test_acc = 100. * correct / total
    print('\n{} set: Accuracy: {}/{}({:.2f}%), Average loss: {:.8f}'.
          format(set_name, correct,  total, test_acc, test_loss))
    return test_acc, test_loss


def train(model, tr_data, val_data,test_data, opt, epoch=10):
    criterion = CrossEntropyLoss()
    loss_list = {'Train': [], 'Validation': []}
    acc_list = {'Train': [0], 'Validation': [0]}
    for i in range(epoch):
        model.train()
        print('Epoch: ', i)
        for data in tqdm(tr_data, position=0,leave=True):
            opt.zero_grad()
            preds = model(data.premise, data.hypothesis)
            loss = criterion(preds, data.label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
        acc, lss = evaluate(model, tr_data, criterion, "Train")
        acc_list['Train'].append(acc)
        loss_list['Train'].append(lss)
        acc, lss = evaluate(model, val_data, criterion, "Validation")
        acc_list['Validation'].append(acc)
        loss_list['Validation'].append(lss)
    evaluate(model, test_data, criterion, 'Test')
    save_graph(acc_list['Train'], acc_list['Validation'], 'Accuracy')
    save_graph(loss_list['Train'], loss_list['Validation'], 'Loss')
    with open('/content/drive/My Drive/DL_4/data/acc_val','wb') as fp:
      fp.write(' '.join([str(i) for i in acc_list['Validation']]))
    with open('/content/drive/My Drive/DL_4/data/loss_val','wb') as fp:
      fp.write(' '.join([str(i) for i in loss_list['Validation']]))


def get_emb_set():
    emb_set = set()
    check_cache = 'data/vocab'
    if os.path.exists(check_cache):
        with open(check_cache, 'rb') as fp:
            return pickle.load(fp)
    for file_path in [EMB_FASTTEXT]:
        with open(file_path, encoding='utf8') as fp:
            for line in tqdm(fp):
                if len(line.split()) != 301:
                    continue
                emb_set.add(line.split()[0])
    with open(check_cache, 'wb') as fp:
        pickle.dump(emb_set, fp)
    return emb_set


def main():
    emb_set = get_emb_set()
    print(f"Embedding size: {len(emb_set)}")
    dt = SNLI_DataLoader(emb_set)
    model = MainModel(dt.get_text_2_id_vocabulary()).to(device)
    opt = Adam(model.parameters(), lr=LR)
    train(model, dt.train_iter, dt.val_iter,dt.test_iter, opt)
    print('x')


if __name__ == '__main__':
    main()
