import os
import pickle

from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from Model import MainModel, EMB_FASTTEXT, EMB_GLOVE
from dataloader import SNLI_DataLoader
from torch import cuda, nn

LR = 0.0007
device = 'cuda' if cuda.is_available() else 'cpu'


def evaluate(model, data_loder, criterion, set_name):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
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


def train(model, tr_data, val_data, opt, epoch=30):
    criterion = CrossEntropyLoss()
    for i in range(epoch):
        model.train()
        print('Epoch: ', i)
        for data in tqdm(tr_data, miniters=500):
            opt.zero_grad()
            preds = model(data.premise, data.hypothesis)
            loss = criterion(preds, data.label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
        adjust_lr(opt, i+1)
        evaluate(model, val_data, criterion, 'Validation')
        # evaluate(model, tr_data, criterion, 'Train')



def adjust_lr(optimizer, epoch):
    lr = LR*0.9**epoch
    print(f"New LR:{lr}")
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_emb_set():
    emb_set = set()
    check_cache = 'data/vocab'
    if os.path.exists(check_cache):
        with open(check_cache, 'rb') as fp:
            return pickle.load(fp)
    for file_path in [EMB_FASTTEXT, EMB_GLOVE]:
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
    train(model, dt.train_iter, dt.val_iter, opt)
    print('x')


if __name__ == '__main__':
    main()
