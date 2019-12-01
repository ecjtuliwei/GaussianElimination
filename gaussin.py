# 调试模式下理论解为[-0.49105816, -0.05088609, 0.36725741]
import numpy as np
from numba import jit
import time

DEBUG = 0  # 调试模式
MODEL = 1   # 用于选择消元方式
# 1: 高斯消元
# 2: 列选主元
# 3: 全选主元
# 4: 高斯若当
n = 5000    # 非测试模式下矩阵大小设置

@jit(nopython=True)     # 加速numpy计算
def gaussin(B, n):
    for k in range(0, n-1):    # 消元的整个过程如下，总共n-1次消元过程。
        # 第K次的消元计算
        print('第', k, '次消元')
        for i in range(k+1, n):
                B[i, :] = B[i, :] - B[i][k] * B[k, :] / B[k][k]
    x = np.zeros(n)  #解的存储数组
    #先计算出最后一个未知数；
    x[n - 1] = B[n - 1][n] / B[n - 1][n - 1]
    #求出每个未知数的值
    for i in range(n-2, -1, -1):
        sum = 0.0
        for j in range(i+1, n):
            sum += B[i][j] * x[j]
        x[i] = (B[i][n] - sum) / B[i][i]
    return x


@jit(nopython=True)
def Col_max(B, n):
    for k in range(n):  # k表示第对k行进行操作
        max = abs(B[k][0])  # 找出列中最大的值
        max_row = k
        for i in range(k, n):
            if abs(B[i][k]) >= max:
                max_row = i
                max = abs(B[i][k])
        #交换行
        temp = B[max_row].copy()
        B[max_row] = B[k]
        B[k] = temp
        for i in range(k+1, n):
                B[i, :] = B[i, :] - B[i][k] * B[k, :] / B[k][k]
        # print('消元 ', k + 1, '\n', B)

    x = np.zeros(n)   #解的存储数组
    #先计算出最后一个未知数；
    x[n-1] = B[n-1][n] / B[n-1][n-1]
    #求出每个未知数的值
    for i in range(n-2, -1, -1):
        sum = 0.0
        for j in range(i+1, n):
            sum += B[i][j] * x[j]
        x[i] = (B[i][n] - sum) / B[i][i]
    return x


@jit(nopython=True)
def All_max(B, n):
    locate = np.arange(n)   # 用于记录未知量位置信息
    for k in range(n-1):    # k表示第对k行进行操作
        #   找出列中最大的值
        max = abs(B[k][0])
        max_row = k
        max_row = k
        for i in range(k, n):
            for j in range(k, n):
                if abs(B[i][j]) >= max:
                    max_row = i
                    max_col = j
                    max = abs(B[i][j])
        #   交换行
        temp = B[max_row].copy()
        B[max_row] = B[k]
        B[k] = temp
        #   交换列
        temp = B[:, max_col].copy()
        B[:, max_col] = B[:, k]
        B[:, k] = temp
        # 记录位置信息
        v = locate[k]
        locate[k] = locate[max_col]
        locate[max_col] = v

        for i in range(k+1, n):
                B[i, :] = B[i, :] - B[i][k] * B[k, :] / B[k][k]
    x = np.zeros(n)   #解的存储数组
    x[n-1] = B[n-1][n] / B[n-1][n-1]
    for i in range(n-2, -1, -1):
        sum = 0.0
        for j in range(i+1, n):
            sum += B[i][j] * x[j]
        x[i] = (B[i][n] - sum) / B[i][i]
    # 交换解的位置
    x1 = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if locate[j] == i:
                x1[i] = x[j]
    return x1


@jit(nopython=True)
def gauss_jordan(B, n):
    for k in range(n):   # k表示第对k行进行操作
        # print('正在对第 ', k, '\t 行进行消元')
        B[k, :] = B[k, :]/B[k, k]   # 将第k行归一化
        for i in range(n):
            if i == k:
                continue
            B[i, :] = B[i, :] - B[i][k] * B[k, :]   #消元操作
    x = np.zeros(n)  # 解的存储数组
    # 先计算出最后一个未知数；
    x[n - 1] = B[n - 1][n] / B[n - 1][n - 1]
    # 求出每个未知数的值
    for i in range(n - 2, -1, -1):
        sum = 0.0
        for j in range(i + 1, n):
            sum += B[i][j] * x[j]
        x[i] = (B[i][n] - sum) / B[i][i]
    return x



def main(n):
    t = time.time()
    if DEBUG == 0:
        print('准备测试数据中')
    #   数据准备
    if DEBUG:   # 调试模式
        n = 3
        B = np.array(
            [[0.001,    2.0,    3.0,    1.0],
             [-1.000,     3.712,  4.623,  2.0],
             [-2.000,     1.072,  5.643,  3.0]])
        x = np.array([-0.49105816, -0.05088609, 0.36725741])
    else:
        # B = np.random.normal(size=(n, n+1))
        B = np.random.normal(size=(n, n + 1))
        # B = 1j * B1
        # B += B1
        # x = np.arange(1, n+1)
        # b = np.zeros(n)
        # for i in range(n):
        #     for j in range(n):
        #         b[i] += B[i][j] * x[j]
        # for i in range(n):
        #     B[i][n] = b[i]

    print('数据准备完成，用时: ', time.time()-t, 's')
    t = time.time()
    # 高斯消元
    if MODEL == 1:
        print('开始高斯消元')
        result = gaussin(B, n)
    #   列选主元
    elif MODEL == 2:
        print('开始列选主元的高斯消元')
        result = Col_max(B, n)
    #   全选主元
    elif MODEL == 3:
        print('开始全选主元的高斯消元')
        result = All_max(B, n)
    else:
        print('开始高斯若当消元')
        result = gauss_jordan(B, n)

    print('方程组的解为： ', result)
    print('总用时 : ', time.time() - t, 's')


if __name__ == "__main__":
    main(n)



