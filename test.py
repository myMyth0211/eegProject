import os
import time

import numpy as np
import torch
from torch.utils import data
from torchvision import transforms

from plot_all import get_data
import net


class TestData(data.Dataset):
    def __init__(self, tester="S1"):
        self.data_root = "./test_data_numpy/data_" + tester + ".npy"
        self.event_root = "./test_data_numpy/event_" + tester + ".npy"
        if os.path.exists(self.data_root) and os.path.exists(self.event_root):
            print("---文件存在---")
            self.data = np.load(self.data_root, allow_pickle=True)
            self.event = np.load(self.event_root, allow_pickle=True)
        else:
            print("---文件不存在---")
            print("----------开始写入数据----------")
            chs = ["B", "D", "G", "L", "O", "Q", "S", "V", "Z"]
            train, event = get_data(tester, chs=chs, turns=[1, 2, 3, 4, 5])
            event = np.asarray(event)
            self.data = []
            self.event = event
            for i_index in range(len(train)):
                for k_index in range(12):
                    start_point = event[i_index, k_index, 1] - event[i_index, 0, 1]
                    tem_train = np.asarray(train[i_index])
                    tem_data = tem_train[start_point:start_point + 150, :]
                    self.data.append(tem_data)
            self.data = np.asarray(self.data)
            np.save(self.data_root, self.data)
            np.save(self.event_root, self.event)
            print("----------数据写入结束----------")

    def __getitem__(self, index):

        return self.data[index, :, :]

    def __len__(self):
        return self.data.shape[0]


def getTestLoader(name):
    t1 = time.time()
    mydata = TestData(name)
    test_loader = data.DataLoader(mydata, batch_size=1, shuffle=True)
    t2 = time.time()
    print("the time of getting data =%f s" % (t2 - t1))
    return test_loader


def test(name):
    device = torch.device("cuda:0")
    test_loader = getTestLoader(name)
    model = net.testNet()
    checkpoint = torch.load("final_net.pth")
    model.load_state_dict(checkpoint["net"])
    model.to(device=device)
    test_out = list()
    for test_data in test_loader:
        test_data = test_data.permute(0, 2, 1).float()
        # test_data = torch.tensor(test_data)
        test_data = test_data.clone().detach()
        test_data = test_data.to(device)

        output = model(test_data)
        test_out.append(1 - output.argmax())
    test_out = np.asarray(test_out, dtype=np.int)
    test_out = test_out.reshape(9, 5, 12)
    return test_out


if __name__ == "__main__":
    result = test("S1")
    print(type(result))
    for i in range(9):
        print(i+1, ":\n")
        print(result[i, :, :])
