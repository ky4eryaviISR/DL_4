from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.functional import F
from tqdm import tqdm
from Model import MainModel
from dataloader import SNLI_DataLoader


def evaluate(model, data, criterion, set_name):
    model.eval()
    test_loss = 0
    correct = 0
    for x, y in data:
        output = model(x)
        test_loss += criterion(output, y).item()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(y.view_as(pred)).sum()
    test_loss /= len(data.dataset)
    test_acc = 100. * correct / len(data.dataset)
    print('{} set: Average loss: {:.4f}, Accuracy: {}/{}({:.0f}%)'.
          format(set_name, test_loss, correct, len(data.dataset), test_acc))
    return test_acc, test_loss


def train(model, tr_data, val_data, opt, epoch=10):
    criterion = CrossEntropyLoss()
    for i in range(epoch):
        model.train()
        for x, y in tqdm(tr_data):
            opt.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            opt.step()

        evaluate(model, tr_data, criterion, 'Train')
        evaluate(model, val_data, criterion, 'Validation')


emb_file = ['data/fasttext/wiki-news-300d-1M.vec',
            'data/glove/glove.6B.50d.txt',
            'GoogleNews-vectors-negative300.bin']


def main():
    dt = SNLI_DataLoader()
    model = MainModel(256, dt.get_text_2_id_vocabulary())
    opt = Adam(model.parameters(), lr=1e-2)

    train(model, dt.train_iter, dt.val_iter, opt)
    print('x')


if __name__ == '__main__':
    main()
