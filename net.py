import torch
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter
#import tensorflow


class testNet(nn.Module):

    def __init__(self):
        super(testNet, self).__init__()
        self.layer1 = nn.Sequential(
            *[nn.Conv1d(in_channels=20, out_channels=20, kernel_size=3, stride=1, padding=1, dilation=2),
              nn.MaxPool1d(kernel_size=2, stride=2),
              nn.BatchNorm1d(20),
              nn.Dropout(0.2),
              nn.ReLU()] * 4)
        self.combine = nn.Conv1d(in_channels=20, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.fc = nn.Linear(in_features=7, out_features=2)
        self.classifier = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.layer1(x)
        feature = self.combine(x)
        feature = torch.squeeze(feature)
        out = self.fc(feature)
        out = self.classifier(out)
        return out


if __name__ == "__main__":
    testnet = testNet()
    print(testnet)
    test_random = np.random.random((1, 20, 150))
    test_out = testnet(torch.tensor(test_random, dtype=torch.float))
    print(test_out.argmax())
    print(test_out)
    dummyinput = torch.rand(1, 20, 150)
    with SummaryWriter("./log", comment="test_net") as w:
        w.add_graph(testnet, (dummyinput,))
