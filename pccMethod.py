import os

import numpy as np
import matplotlib.pyplot as plt

import filter
from write_data import getRightChar


def getP300(name):
    data_dir = "data_numpy/data_" + name + ".npy"
    label_dir = "data_numpy/label_" + name + ".npy"

    p300_dir = "p300_data/p300_" + name + ".npy"
    if os.path.exists(p300_dir):
        result = np.load(p300_dir)
        print("源文件已存在.\n")
    else:
        data = np.load(data_dir)
        print(data.shape)
        label = np.load(label_dir)
        print(label.shape)

        result = np.zeros(150, dtype=np.float)
        N = label.size() / 2
        for i in range(N):
            if label[i][0] == 1:
                data_sum = np.sum(data[i], axis=1)
                result += data_sum
        result = result
        np.save(p300_dir, result)
    return result


def cal_pccs(x, y, n=150):
    """
    warning: data format must be narray
    :param x: Variable 1
    :param y: The variable 2
    :param n: The number of elements in x
    :return: pccs
    """
    sum_xy = np.sum(np.sum(x * y))
    sum_x = np.sum(np.sum(x))
    sum_y = np.sum(np.sum(y))
    sum_x2 = np.sum(np.sum(x * x))
    sum_y2 = np.sum(np.sum(y * y))
    pcc = (n * sum_xy - sum_x * sum_y) / np.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
    return pcc if pcc > 0 else 0


def plot_p300(p300_data):
    plt.plot(p300_data)
    plt.show()


if __name__ == "__main__":
    s2_p300 = getP300("S2")
    s2_x = np.load("./test_data_numpy/data_S2.npy")
    score1 = []
    score2 = []
    num = 1
    for i in range(12):
        data_pcc = s2_x[i + 12 * num, :, :]
        data_pcc = np.sum(data_pcc, axis=1)
        pcc_score = cal_pccs(data_pcc, s2_p300)
        if i < 6:
            score1.append(pcc_score)
        else:
            score2.append(pcc_score)
    loc1 = score1.index(max(score1)) + 1
    loc2 = score2.index(max(score2)) + 7

    print("location1:", loc1)
    print("location2:", loc2)
    try:
        right_char = getRightChar(loc1, loc2)
        print("_____________")
        print("test char:", right_char)
    except:
        print("out of index")

    # plot_p300(s2_p300)
    # filter_data = filter.butter_filter(s2_p300)
