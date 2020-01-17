from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.functional import F
from tqdm import tqdm
from Model import MainModel
from dataloader import SNLI_DataLoader
from torch import cuda

LR = 0.01
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
    print('{} set: Accuracy: {}/{}({:.0f}%), Average loss: {:.8f}'.
          format(set_name, correct,  total, test_acc, test_loss))
    return test_acc, test_loss


def train(model, tr_data, val_data, opt, epoch=10):
    criterion = CrossEntropyLoss()
    for i in range(epoch):
        model.train()
        for data in tqdm(tr_data):
            opt.zero_grad()
            preds = model(data.premise, data.hypothesis)
            loss = criterion(preds, data.label)
            loss.backward()
            opt.step()

        evaluate(model, val_data, criterion, 'Validation')

def adjust_lr(optimizer, epoch):
    lr = LR*(1 - epoch/20)**0.9
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

emb_file = ['data/fasttext/wiki-news-300d-1M.vec',
            'data/glove/glove.6B.50d.txt',
            'GoogleNews-vectors-negative300.bin']


def main():
    dt = SNLI_DataLoader()
    model = MainModel(emb_file, dt.get_text_2_id_vocabulary()).to(device)
    opt = Adam(model.parameters(), lr=0.001)

    train(model, dt.train_iter, dt.val_iter, opt)
    print('x')


if __name__ == '__main__':
    main()
