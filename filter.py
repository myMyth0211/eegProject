import numpy as np
from plot_all import get_data
from scipy import signal
from matplotlib import pyplot as plt


# 此函数用于实现脑电数据的滤波，提取波形
def butter_filter(data, n=150):
    fs = 250
    t = np.linspace(0, (n - 1) / fs, n)

    Wn = [0.5 * 2 / fs, 8 * 2 / fs]
    [b, a] = signal.butter(4, Wn, 'bandpass')
    filtered = signal.filtfilt(b, a, data)
    # 数据可视化
    plt.plot(t, filtered, 'g')
    plt.xlim([0, n / 250])
    plt.grid()
    plt.title("the result of filter")
    plt.show()

    return filtered


if __name__ == "__main__":
    train_data, event_data = get_data("S1", ["B"], [2])
    train_data, event_data = np.asarray(train_data[0]), np.asarray(event_data[0])
    # print(train_data.shape)
    # print(event_data.shape)

    train_data = np.sum(train_data, axis=1)

    # data1 = np.sum(data1, axis=1)
    i = 10  # 在一轮实验中的次序
    new_data = train_data[event_data[i, 1] - event_data[0, 1]:event_data[i, 1] - event_data[0, 1] + 150]
    print(new_data.size)
    filtered_data = butter_filter(data=new_data)  # 对脑电波进行巴特沃斯滤波
