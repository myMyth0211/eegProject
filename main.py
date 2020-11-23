import torch

import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
import torch.cuda
from torch.autograd import Variable

import net
from write_data import *

epoches = 100
lr = 0.001
momentum = 0.8
weight_decay = 0.0001

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = net.testNet()
model.to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
criterion = nn.BCELoss()
train_loader = get_loader("S1")
print("------------开始训练------------")

train_loss_total = []
train_acc_total = []
best_score = 0
for epoch in range(epoches):
    model.train()
    train_loss = 0.
    train_acc = 0.

    for data, label in train_loader:

        data = data.permute(0, 2, 1).float()
        data = data.cuda()
        label = label.cuda()

        optimizer.zero_grad()
        predicted = model(data)
        loss = criterion(predicted, label[0].float())

        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        index1 = predicted.argmax()
        index2 = label[0].argmax()
        if index1 == index2:
            train_acc += 1

    train_loss /= 720
    train_loss_total.append(train_loss)
    train_acc /= 720
    train_acc_total.append(train_acc)
    if train_acc > best_score:
        best_score = train_acc
        state = {"net": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}
        torch.save(state, "final_net.pth")
    print("eporch={}:the accuracy : {:.2f}%,train loss:{:.4f}".format(epoch + 1, train_acc * 100, train_loss))
train_acc_total = np.asarray(train_acc_total)
train_loss_total = np.asarray(train_loss_total)

np.save("train_loss.npy", train_loss_total)
np.save("train_acc.npy", train_acc_total)

print(train_acc_total)
print(train_loss_total)

print("------------结束训练------------")
