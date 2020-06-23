import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("pokemon.csv")
data = df[['cp', 'cp_new']].values
test_size = 0.33
epsilon = 0.001


def compute_error(b, w, points):
    """
    计算平均损失
    """
    x = points[:, 0]
    y = points[:, 1]
    return np.mean((y - (w * x + b))**2)


def step_gradient(b, w, points, learning_rate):
    """
    计算新的w和b的值
    """
    x = points[:, 0]
    y = points[:, 1]
    error = y - (w * x + b)
    b_gradient = 2 * np.mean(error)
    w_gradient = 2 * np.mean(error * (-x))
    b_new = b - (learning_rate * b_gradient)
    w_new = w - (learning_rate * w_gradient)
    return b_new, w_new


def main():
    train_data = data[:-int(test_size * len(data))]
    test_data = data[-int(test_size * len(data)):]
    plt.figure(figsize=(20, 10))
    plt.axis([0, 500, 0, 780])
    plt.ylabel("cp_new")
    plt.xlabel("cp")
    plt.scatter(train_data[:, 0], train_data[:, 1], label="train")
    plt.scatter(test_data[:, 0], test_data[:, 1], label='test')
    plt.legend()

    w = 0.0
    b = 0.0
    learning_rate = 0.00001
    x = np.arange(500)
    y = w * x + b
    for i in range(200):
        np.random.shuffle(data)
        train_data_ = data[:-int(0.1 * len(data))]
        test_data_ = data[-int(0.1 * len(data)):]
        train_error = compute_error(b, w, train_data_)
        test_error = compute_error(b, w, test_data_)
        print("{}. 训练平均误差:".format(i + 1), train_error, "测试平均误差:", test_error)
        if (train_error > epsilon):
            y = w * x + b
            b, w = step_gradient(b, w, train_data_, learning_rate)
            print("{}. w:{} b:{}".format(i + 1, w, b))
    print("测试平均误差:", compute_error(b, w, test_data))
    plt.plot(x, y)
    plt.show()


if __name__ == "__main__":
    main()