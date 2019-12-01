import numpy as np
from numba import jit
import time
n = 5000    # 设置矩阵大小


@jit(nopython=True)
def gauss_jordan(B, n):
    for k in range(n):   # k表示第对k行进行操作
        print('正在对第 ', k, '\t 行进行消元')
        B[k, :] = B[k, :]/B[k, k]   # 将第k行归一化
        for i in range(n):
            if i == k:
                continue
            B[i, :] = B[i, :] - B[i][k] * B[k, :]   #消元操作
    return B


def main():
    print('正在生成测试矩阵，请稍等\n......')
    B = np.random.normal(size=(n, n*2))         # 生成随机矩阵n*（n+1）
    for i in range(n):
        for j in range(n, n*2):
            if i == j-n:
                B[i][j] = 1.0
            else:
                B[i][j] = 0.0           # 将后半部分改为单位阵
    print('数据准备完毕，开始消元操作\n')
    t = time.time()
    gauss_jordan(B, n)
    print('总共用时： ', time.time()-t, 's')


if __name__ == "__main__":
    main()


