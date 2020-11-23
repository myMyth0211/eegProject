#!/usr/bin/evn python
# -*- coding:utf-8 -*-
import xlrd
import xlwt
import matplotlib.pyplot as plt
import numpy as np


# 获取对应被试者的不同轮次的脑电图
def get_data(person, chs, turns, type="train"):
    """
    :param person: 被试的编号，范围从S1-S5
    :param chs: 脑电图测试字符数,可取【BDGLOQSVZ479】
    :param turns: 测试的轮次，取值范围为1~5
    :return:
    """
    allChar = "BDGLOQSVZ479"
    for turn in turns:
        if not isinstance(turn, int):
            print("请输入一个整数轮次.\n")
            exit()
        if turn < 1 or turn > 5:
            print("轮次应该大于0而小于6.\n")
            exit()
    indexs = []
    for ch in chs:
        index = allChar.find(ch)
        indexs.append(index)
        if index == -1:
            print("测试数据中没有相应字符.\n")
            exit()

    filename1 = "data/" + person + "/" + type + "_data.xlsx"
    filename2 = "data/" + person + "/" + type + "_event.xlsx"
    train_file = xlrd.open_workbook(filename1)
    event_file = xlrd.open_workbook(filename2)

    final_train_data = []
    final_event_data = []
    for index in indexs:
        for turn in turns:
            train_data = train_file.sheets()[index]
            event_data = event_file.sheets()[index]

            train_size = [train_data.nrows, train_data.ncols]
            event_size = [event_data.nrows, event_data.ncols]

            # 获取对应轮次的事件流
            event_target = np.zeros((12, 2), dtype=int)
            for i in range(event_size[1]):
                cols = event_data.col_values(i)
                event_target[:, i] = cols[13 * (turn - 1) + 1:13 * turn]

            # 获取对应轮次的脑电数据
            train_target = np.zeros((event_target[11, 1] - event_target[0, 1] + 150, 20))
            for j in range(train_size[1]):
                cols = train_data.col_values(j)
                train_target[:, j] = cols[event_target[0, 1]:event_target[11, 1] + 150]

            final_train_data.append(train_target)
            final_event_data.append(event_target)
    return final_train_data, final_event_data


# 函数入口，用于绘制指定测试者的相应字符的各轮脑电图
def plot_all_channel(person, chars, turns):
    train, _ = get_data(person, chars, turns, type="test")  # 获取脑电波数据

    # 数据可视化，绘制20个通道的折线图
    print(chars[0])
    plt.plot(train[0])
    plt.title("all eeg for one test:" + chars[0])
    plt.show()

    # 绘制通道求和的结果
    plt.plot(np.sum(train[0], axis=1))
    plt.title("the result of sum:")
    plt.show()


def plot_loss(data):
    plt.plot(data, 'r')
    plt.title("Loss for 100 epoch")
    plt.grid()
    plt.show()


def plot_acc(data):
    plt.plot(data)
    plt.title("accuracy for train data")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    acc_data = np.load("train_acc.npy")
    plot_acc(acc_data)
