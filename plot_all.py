#!/usr/bin/evn python
# -*- coding:utf-8 -*-
import xlrd
import xlwt
import matplotlib.pyplot as plt
import numpy as np


# 获取对应被试者的不同轮次的脑电图
def get_data(person="S1", ch="B", turn=1):
    """
    :param person: 被试的编号，范围从S1-S5
    :param ch: 脑电图测试字符数,可取【BDGLOQSVZ479】
    :param turn: 测试的轮次，取值范围为1~5
    :return:
    """
    allChar = "BDGLOQSVZ479"
    if not isinstance(turn, int):
        print("请输入一个整数轮次.\n")
        exit()
    if turn < 1 or turn > 5:
        print("轮次应该大于0而小于6.\n")
        exit()
    index = allChar.find(ch)
    if index == -1:
        print("测试数据中没有相应字符.\n")
        exit()

    filename1 = "data/" + person + "/train_data.xlsx"
    filename2 = "data/" + person + "/train_event.xlsx"
    train_file = xlrd.open_workbook(filename1)
    event_file = xlrd.open_workbook(filename2)

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
    train_target = np.zeros((event_target[11, 1] - event_target[0, 1] + 151, 20))
    for j in range(train_size[1]):
        cols = train_data.col_values(j)
        train_target[:, j] = cols[event_target[0, 1] - 1:event_target[11, 1] + 150]

    return train_target


# 函数入口，用于绘制指定测试者的相应字符的各轮脑电图
def main(person="S1", ch="B", turn=1):
    train = get_data(person, ch, turn)  # 获取脑电波数据

    # 数据可视化，绘制20个通道的折线图
    plt.plot(train)
    plt.title("all eeg for one test")
    plt.show()

    # 绘制通道求和的结果
    plt.plot(np.sum(train, axis=1))
    plt.title("the result of sum")
    plt.show()


if __name__ == "__main__":
    main(person="S1", ch="9", turn=5)
