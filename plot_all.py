#!/usr/bin/evn python
# -*- coding:utf-8 -*-
import xlrd
import xlwt
import matplotlib.pyplot as plt
import numpy as np


# 此函数用于绘制指定测试者的相应字符的各轮脑电图
def main(person="S1", ch="B", trun=1):
    allChar = ["B", "D", "G", "L", "O", "Q", "S", "V", "Z", "4", "7", "9"]
    if not isinstance(trun, int):
        print("请输入一个整数轮次.\n")
        exit()
    if trun < 1 or trun > 5:
        print("轮次应该大于0而小于6.\n")
        exit()
    try:
        index = allChar.index(ch)  # 获取字符对应的sheet索引
    except:
        print("测试数据中没有此字符.\n")
    else:

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
            event_target[:, i] = cols[13 * (trun - 1) + 1:13 * trun]

        # 获取对应轮次的脑电数据
        train_target = np.zeros((event_target[11, 1] - event_target[0, 1] + 151, 20))
        for j in range(train_size[1]):
            cols = train_data.col_values(j)
            train_target[:, j] = cols[event_target[0, 1] - 1:event_target[11, 1] + 150]

        # 数据可视化，绘制20个通道的折线图
        #        for k in range(20):
        #           plt.subplot(4, 5, k + 1)
        #           plt.plot(train_target[:, k])
        plt.plot(train_target)
        plt.title("all eeg for one test")
        plt.show()

        #绘制通道求和的结果
        plt.plot(np.sum(train_target, axis=1))
        plt.title("the result of sum")
        plt.show()


if __name__ == "__main__":
    main(person="S3", ch="9", trun=5)
