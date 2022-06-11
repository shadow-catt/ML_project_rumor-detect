import torch
from lstm_model import LSTM_Model
from dataset import RummorDataset
import config as config
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from dataset import  get_dataloader
from torch.nn import CrossEntropyLoss
from utils import  plot_learning_curve
device = config.device

def train(epoch,model,loss_fn,optimizer,train_dataloader):
    model.train()
    loss_list = []
    train_acc = 0
    train_total = 0
    loss_fn.to(device)
    bar = tqdm(train_dataloader, total=len(train_dataloader))  # the progress bar
    for idx, (input, target) in enumerate(bar):
        input = input.to(device)
        target = target.to(device)
        output = model(input)
        loss =loss_fn(output, target)
        optimizer.zero_grad()
        loss.backward()
        loss_list.append (loss.cpu().item())
        optimizer.step()
        # 准确率
        output_max = output.max (dim=-1)  # Returns the maximum value and corresponding index
        pred = output_max[-1]  # Index of maximum value
        train_acc += pred.eq (target).cpu ().float ().sum ().item ()
        train_total += target.shape[0]
    acc = train_acc / train_total
    print("train epoch:{}  loss:{:.6f} acc:{:.5f}".format(epoch, np.mean(loss_list),acc))
    return acc,np.mean(loss_list)


def test(model,loss_fn,test_dataloader):
    model.eval()
    loss_list = []
    test_acc=0
    test_total=0
    loss_fn.to (device)
    nb_classes = 2
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        for input, target in test_dataloader:
            input = input.to(device)
            target = target.to(device)
            output = model(input)
            _, preds = torch.max(output, 1)
            for t, p in zip(target.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            loss = loss_fn(output, target)
            loss_list.append(loss.item())
            # accuracy
            output_max = output.max(dim=-1) # Returns the maximum value and corresponding index
            pred = output_max[-1]  # the index of the meximum value
            test_acc+=pred.eq(target).cpu().float().sum().item()
            test_total+=target.shape[0]
        acc=test_acc/test_total
        print('confusion_matrix:\n',confusion_matrix.numpy())

        # To get the per-class accuracy: precision
        precision = (confusion_matrix.diag() / confusion_matrix.sum(1)).mean()

        recall = (confusion_matrix.diag() / confusion_matrix.sum(0)).mean()

        f1 = (2 * precision * recall / (precision + recall)).mean()


        print("precision:", precision.numpy(),"\trecall:",recall.numpy(),"\tf1:", f1.numpy())
        print("test loss:{:.6f},acc:{}".format(np.mean(loss_list), acc))
    return acc,np.mean(loss_list)



if __name__ == '__main__':
    model = LSTM_Model().to(device)
    count_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {count_parameters:,} trainable parameters')
    optimizer = optim.Adam(model.parameters(), lr=config.lr)  # adaptive gradiant optimizer
    train_dataloader = get_dataloader(model='train')
    test_dataloader = get_dataloader(model='test')
    loss_fn=CrossEntropyLoss()  ## cross entropy loss
    best_acc=0
    early_stop_cnt=0
    train_loss_list=[]
    train_acc_list=[]
    test_loss_list=[]
    test_acc_list=[]
    for epoch in range(config.epoch):
        train_acc,train_loss=train(epoch,model,loss_fn,optimizer,train_dataloader)
        test_acc,test_loss=test(model,loss_fn,test_dataloader)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        if test_acc>best_acc:
            best_acc=test_acc
            torch.save(model.state_dict(), 'model/model.pkl')   # save the model
            print("save model,acc:{}".format(best_acc))
            early_stop_cnt=0
        else:
            early_stop_cnt+=1
        if early_stop_cnt>config.early_stop_cnt:
            break
    plot_learning_curve(train_loss_list,test_loss_list,'loss')
    plot_learning_curve(train_acc_list, test_acc_list,'accuracy')


