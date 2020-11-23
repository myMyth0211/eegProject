import os
import time

import numpy as np
import torch
from torch.utils import data
from torchvision import transforms

from plot_all import get_data


class MyData(data.Dataset):
    def __init__(self, tester="S1"):
        self.data_root = "./data_numpy/data_" + tester + ".npy"
        self.label_root = "./data_numpy/label_" + tester + ".npy"
        if os.path.exists(self.data_root) and os.path.exists(self.label_root):
            print("---文件存在---")
            self.data = np.load(self.data_root, allow_pickle=True)
            self.label = np.load(self.label_root, allow_pickle=True)
        else:
            print("---文件不存在---")
            print("----------开始写入数据----------")
            chs = ["B", "D", "G", "L", "O", "Q", "S", "V", "Z", "4", "7", "9"]
            train, event = get_data(tester, chs=chs, turns=[1, 2, 3, 4, 5])
            event = np.asarray(event)
            self.data = []
            self.label = []
            for i_index in range(len(train)):
                for k_index in range(12):
                    start_point = event[i_index, k_index, 1] - event[i_index, 0, 1]
                    tem_train = np.asarray(train[i_index])
                    tem_data = tem_train[start_point:start_point + 150, :]
                    self.data.append(tem_data)
                    if judegeLabel(event[i_index, k_index, 0], chs[i_index // 5]):
                        self.label.append([1, 0])
                    else:
                        self.label.append([0, 1])

            np.save(self.data_root, np.asarray(self.data))
            np.save(self.label_root, np.asarray(self.label))
            print("----------数据写入结束----------")

    def __getitem__(self, index):

        return self.data[index, :, :], self.label[index]

    def __len__(self):
        return len(self.label)


def get_loader(name):
    t1 = time.time()
    mydata = MyData(name)
    tranform = transforms.Compose(transforms.ToTensor())
    train_loader = data.DataLoader(mydata, batch_size=1, shuffle=True)
    t2 = time.time()
    print("the time of getting data =%f s" % (t2 - t1))
    return train_loader


label_index = ["A", "B", "C", "D", "E", "F",
               "G", "H", "I", "J", "K", "L",
               "M", "M", "O", "P", "Q", "R",
               "S", "T", "U", "V", "W", "X",
               "Y", "Z", "1", "2", "3", "4",
               "5", "6", "7", "8", "9", "0"]


def judegeLabel(number, target_str):
    index = label_index.index(target_str)
    if number == index // 6 + 1 or number == index % 6 + 7:
        return True
    else:
        return False


def getRightChar(num1, num2):
    if num1 < 7 and num2 > 6:
        return label_index[(num1 - 1) * 6 + (num2 - 7)]
    else:
        print("ERROR:number is out of index.")
        return -1


if __name__ == "__main__":
    get_loader("S1")
    get_loader("S2")
    get_loader("S3")
    get_loader("S4")
    get_loader("S5")
