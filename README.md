# **高斯消元法**

摘  要：本文档介绍了高斯消元法、列选主元的高斯消元、全选主元的高斯消元法、高斯-若当消元求逆的原理及实现过程，并借助Numba实现了5000*5000矩阵的快速消元求解。分析对比了三种消元方法的优缺点。

关键词：高斯消元、列选主元、全选主元、高斯-若当

 

# **Gaussian Elimination**

**Abstract:** This paper introduces the principle and implementation process of Gaussian elimination, Elimination with Maximal Column pivot，Elimination with Maximal pivot,and Gauss-Jordan elimination.with the help of Numba, the matrix of 5000*5000 can be realized in a short time. This paper also analyze and compare the advantages and disadvantages of the three elimination methods.

**Key word:** Gaussian elimination ,Gauss-Jordan

## **1** **引言**

​	科学研究或者工程建设当中经常会遇到许多变量呈现线性关系的情况, 通常的做法是将这些问题线性化, 转化为线性方程组。因此线性方程组的高效求解格外重要。矩阵分解的线性线性方程的求解方法有很多。一般的, 我们通过一些基本定理再经过转化进而简化求解, 而在计算机中应用更为常见的是结合线性方程组系数以及常数形成矩阵, 再利用矩阵的特点和性质来求解。本文中我们主要讨论高斯消元法、列选主元的高斯消元法、全选主元的高斯消元法求解线性方程组及高斯-若当法求矩阵的逆。

## **2** **算法原理及实现过程**

### 2.1高斯消元法

​	设有线性方程组



|      | ![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpscaFfQB.png) | (2.1.1) |
| ---- | ---------------------------------------------------- | ------- |

​	或写为矩阵形式

|      | ![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsMs9myY.png) | (2.1.2) |
| ---- | ---------------------------------------------------- | ------- |

简记为![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsqlxxgl.jpg)，要求解线性方程组就是要求出![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsmVAJYH.jpg)。

​	高斯消元法的基本思想是通过矩阵的初等变换逐次把系数矩阵**A**化为上三角或下三角矩阵，再用回代的方法求出方程组的解。高斯消元过程中选取矩阵对角元素作为主元进行消元操作，如果对角元素出现0，则消元过程无法进行。

​	以下为高斯消元法的过程：

​	将方程组(2.1)记为![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpscSJXG4.jpg)，其中

![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsy7Ddpr.png)

(1) 第一步(k = 1).

设![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wps07Ov7N.jpg)，首先计算乘数

![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wps4HCPPa.png)

用![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsIerbyx.jpg)乘方程组(2.1)的第一个方程，加到第i个![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsIcOygU.jpg)方程上,消去方程组(2.1.1)的从第2个方程到第n个方程中的未知数![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsSxUXYg.jpg),得到与方程组(2.1)等价的线性方程组

|      | ![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsEAdpHD.png) | (2.1.3) |
| ---- | ---------------------------------------------------- | ------- |

简记为![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsYqpTp0.jpg),其中![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsiR5o8m.jpg)，![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wps8poWQJ.jpg)的元素计算公式为

|      | ![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsUsvvz6.jpg) | (2.1.4) |
| ---- | ---------------------------------------------------- | ------- |

(2) 第k次消元![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpssiN6ht.jpg).

设上述第1步,![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wps6yHJ0P.jpg),第k-1步消元过程计算已完成,即已计算好与方程组(2.1.1)等价的线性方程组

|      | ![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsU61oJc.png) | (2.1.5) |
| ---- | ---------------------------------------------------- | ------- |

简记为![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsiLa8rz.jpg),设![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wps8nTSaW.jpg),计算乘数

![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsYQmFTi.png)

用![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsIaFtCF.jpg)乘方程组(2.1.5)的第k个方程，加到第i个![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsmvDjl2.jpg)方程上,消去从第k+1个方程到第n个方程中的未知数![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsKvlb4o.jpg),得到与方程组(2.1.1)等价的线性方程组![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsCbF4ML.jpg).![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsGhBZv8.jpg),![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsOJ1Vev.jpg)元素的计算公式为

![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpskClUXR.png)

显然![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsg6TUGe.jpg)中从第1行到第k行与![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsur9WpB.jpg)相同.

(3) 继续上述过程,且设![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsIS708X.jpg),直到完成第n-1步消元计算,最后得到

与原方程组等价的简单方程组![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsIGX6Rk.jpg),即

|      | ![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsmPXeBH.png) | (2.1.6) |
| ---- | ---------------------------------------------------- | ------- |

由方程组(2.1.1)化为方程组(2.1.6)的过程称为消元过程.

如果![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsIg8pk4.jpg)是非奇异矩阵,且![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsONZC3q.jpg),求解三角矩阵线性方程组(2.1.6)得到求解公式

|      | ![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsGLaSMN.png) | (2.1.7) |
| ---- | :--------------------------------------------------- | ------- |

方程组(2.1.6)的求解过程(2.1.7)称为回代过程.

 

**2****.2****列选主元的高斯消元法**

​	在高斯消元法中,对第k行消元过程中需要将其余行除以对角线元素![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsGzYawa.jpg)，每次消元用作除数的元素叫做主元，如果主元为0，则消元无法继续进行。

​	列选主元的高斯消元法通过找寻主元所在列及主元列之后行元素中绝对值最大的元素并通过初等行变换将最大元素换到主元的位置上再进行消元操作。

​	列选主元的实现过程如下

(1) 构造增广矩阵![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsGdTvfx.jpg)，首先在![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsaNlSYT.jpg)的第一列中选取绝对值最大的元素作为主元素,

例如

![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpssOHgIg.png)

然后交换**B**的第1行与第![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsO58GrD.jpg)行，经过第1次消元得到

![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsQWb9a0.jpg) 

(2) 重复上述过程，设已完成第k-1步的选主元素，交换两行及消元计算，![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wps4ncDUm.jpg)化为

|      | ![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsgYF9DJ.png) | (2.2.1) |
| ---- | ---------------------------------------------------- | ------- |

其中![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wps64zKn6.jpg)的元素仍记为![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpso6bn7s.jpg)，![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsQin1QP.jpg)的元素仍记为![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsqv8GAc.jpg).

(3) 第k步选主元素,即确定![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsQYopkz.jpg),使

![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsG7n93V.png)

交换![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsyxdVNi.jpg)第k行与![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpskuDIxF.jpg)(![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsS7Exh2.jpg))行的元素,再进行消元计算,最后将原线性方程组化为

|      | ![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsyxPo1o.png) | (2.2.2) |
| ---- | ---------------------------------------------------- | ------- |

回代求解的到**x****.**

 

**2****.3****全选主元的高斯消元法**

​	在列选主元的基础上，全选主元把主元的搜索范围由主元所在列拓展到了全部未消元的元素上，通过行变换和列变换将矩阵化为上三角或下三角矩阵,再通过回代求解。

​	全选主元的实现过程如下

(1) 构造增广矩阵![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsg6xjLL.jpg)，首先在![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpscQQfv8.jpg)的所有元素中选取绝对值最大的元素作为主元

素,例如

![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsQeVdfv.png)

然后交换**B**的第1行与第![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsES0dZR.jpg)行，交换**B**的第1列与第![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsQcUfJe.jpg)列，经过第1次消元得到

![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsmvvjtB.jpg) 

​	如果进行了列变换,则将列变换记录到**x**的位置信息里.

(2) 重复上述过程，设已完成第k-1步的选主元素，交换两行及消元计算，![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsaVfpdY.jpg)化为

|      | ![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsQazxXk.png) | (2.3.1) |
| ---- | ---------------------------------------------------- | ------- |

其中![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsqbaKHH.jpg)的元素仍记为![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsYAsYr4.jpg)，![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsyBnecr.jpg)的元素仍记为![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wps6kUvWN.jpg).

(3) 第k步选主元素,即确定![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsA0jPGa.jpg),使

![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsm9larx.png)

交换![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsECixbU.jpg)第k行与![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsu5LVVg.jpg)(![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsCLRlGD.jpg))行的元素, 交换![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsE0uNq0.jpg)第k列与![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsQKHgbn.jpg)(![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpseDvLVJ.jpg))列的元素,再进行消元计算,最后将原线性方程组化为

|      | ![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsmBsiG6.png) | (2.3.2) |
| ---- | ---------------------------------------------------- | ------- |

回代求解的到**x****1.**通过记录的位置信息交换**x****1**的元素的到**x****.**

 

**2****.4****高斯-若当消元法**

​	通过对增广矩阵进行行变换和列变换我们可以将矩阵转换为便于求解的三角矩阵，如果在系数矩阵后增加一个同样大小的单位矩阵构造一个增广矩阵，通过对增广矩阵的初等变换将系数矩阵部分化为单位阵，则原来的单位阵部分就是系数矩阵的逆阵。

​	高斯若当消元方法和高斯消元、列主元、全主元消元方法一致，只是消元的结果前者是化为三角阵，后者是化为单位阵。高斯若当消元可以采用前者的方法，只是在对第k次消元这一步不只消去第k行以后的元素，而是将第k列所有除了主元的元素全部消去，并将主元归一化。

​	列选主元的高斯-若当消元法实现过程如下

(1) 	构造增广矩阵![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsynsTqt.jpg)，首先在![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wps0b9vbQ.jpg)的第一列中选取绝对值最大的元素作为主元素,

例如

![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsEcFaWc.png)

然后交换**B**的第1行与第![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wps0nxRGz.jpg)行，经过第1次消元得到

![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpseh2zrW.jpg) 

(2) 重复上述过程，设已完成第k-1步的选主元素，交换两行及消元计算，![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpscarkcj.jpg)化为

|      | ![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsUaM7WF.png) | (2.4.1) |
| ---- | ---------------------------------------------------- | ------- |

其中![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsko0ZH2.jpg)的元素仍记为![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wps8pQTsp.jpg)，![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpscufPdM.jpg)的元素仍记为![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpscuhMY8.jpg).

(3) 第k步选主元素,即确定![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wps8X9KJv.jpg),使

![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsGtFLuS.png)

交换![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsSH1Nff.jpg)第k行与![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wps6K1R0B.jpg)(![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpswuCXLY.jpg))行的元素,再进行消元计算,最后将增广矩阵化为

|      | ![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpswqk5wl.png) | (2.4.2) |
| ---- | ---------------------------------------------------- | ------- |

![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpswoBfiI.jpg)即为矩阵**A**的逆矩阵。

​	若构建增广矩阵时不是采用单位矩阵，而是如同高斯消元时将方程组的右边加到系数矩阵后，采用高斯-若当的方法方法对增广矩阵消元后的到的最后一列就是方程组的解

## **3** **程序实现**

**3****.1** **实验平台**

| CPU      | AMD Ryzen 5 2600X Six-Core 4.1GHz |
| -------- | --------------------------------- |
| RAM      | DDR4 2400 32G(8G*4)               |
| System   | Windows 10 专业版                 |
| Software | Pycharm、python3.7                |

 

**3****.2** **测试数据准备**

​	在程序实现过程中，首先需要创建一个用于计算的矩阵，同样用于测试。在本次实验中，通过创建随机的系数矩阵和一组解x，计算得到b来创建一个已知所有元素的方程组用于测试，将A和b用于计算，得到的结果与x进行对比来验证算法的准确性。例如

![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wps0div34.png)

![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsGfEOOr.png)

通过![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsMSY9zO.png)得到![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsCvfxlb.png)

![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsWBdW6x.png)

构建增广矩阵为

![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsKzroSU.png)

对B进行消元求解后得到的解与![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsClSUDh.png)对比验证算法是否准确

 

**3****.3** **实现过程**

​	本次实验用Python语言实现，所有的矩阵和向量都用numpy.array实现，算法通过计算公式实现，用一个大循环来对第k行进行操作，在高斯消元法中，每个大循环内先从第k+1行遍历剩余行的第k列元素得到用于消元的系数，再用剩余行每个元素乘以对应系数减去第k行进行消元操作。其余方法结构一致，只是多了选主元和进行初等变换的过程，具体过程不再详述。

​	如果从计算速度来说，使用Python进行科学计算本身不是一个明智的选择，因为Python是解释性语言，代码要一行一行由解释器进行解释再执行，运行的速度较慢，特别是当使用了5000*5000这样的大矩阵时，python需要的时间太久。如下图(3.3.1)是高斯消元过程矩阵从100*100到300*300时计算用时，计算300*300的矩阵需要12秒，且算法的时间复杂度为![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wps6yfupE.jpg)，计算5000*5000矩阵需要用时估算为20个小时。

计算大矩阵时python有点力不从心，所需的时间太长了。此时第一个思路是使用C++，使用C++实现了高斯消元算法，测试了计算时间，相同阶数下比python的用时更短，但时间复制度依旧为![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wps0dM5a1.jpg)，测试得到的曲线上升趋势比python的更快，估算得到计算5000*5000矩阵的用时超过一天。依旧不宜使用。

![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpscc6IWn.jpg)![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wps0JioIK.jpg) 

​	最后程序选择了用python写，通过numba库的jit模块，可以将python内numpy类型的计算加速，通过将代码转化为机器码编译后运行，实际测试大大加速了程序的运行。

**3****.4** **Numba加速**

​	numba是一个用于编译Python数组和数值计算函数的编译器，这个编译器能够大幅提高直接使用Python编写的函数的运算速度。

numba使用LLVM编译器架构将纯Python代码生成优化过的机器码从而加快运行速度。

在实际的测试时，分别收集了计算N从100到1100的用时，如下图所示，python使用了numb.jit装饰器后，计算用时比c++代码快了一个数量级。此时程序依旧是使用cpu单核在运行，这种模式下导入jit(Just-in-Time)装饰器只需在原来的代码中加一句话，如果使用GPU来运行计算，需要对代码进行全部重写，使用特定的语句实现运算，这种模式下如果成功运行在GPU上，则运算速度还会有一个数量级的提升。

![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsCjP6t7.jpg) 

 

## **4** **分析对比**

**4****.1计算结果分析**

​	测试矩阵为

![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsSLNRfu.png)

在限定保留4位小数的情况下，三种算法的到的计算结果如下

| 消元方法 | ![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsqhHF1Q.jpg) | ![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsC9jvNd.jpg) | ![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsIWRmzA.jpg) |
| -------- | ---------------------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------- |
| 高斯消元 | -0.4                                                 | -0.0503                                              | 0.367                                                |
| 列选主元 | -0.4902                                              | -0.0511                                              | 0.3676                                               |
| 全选主元 | -0.4903                                              | -0.051                                               | 0.3675                                               |
| 理论解   | -0.4904                                              | -0.05104                                             | 0.3675                                               |

​	每次计算时采用的主元为

| 消元方法 | 第一次消元-主元 | 第二次消元-主元 | 第三次消元-主元 |
| -------- | --------------- | --------------- | --------------- |
| 高斯消元 | 0.001           | 2003.712        | 6.0118          |
| 列选主元 | -2.0            | -3.176          | 1.868           |
| 全选主元 | -5.643          | 2.8338          | 0.742           |

由计算结果可见，当出现比较小的元素当作主元时，计算的结果经过四舍五入后会产生很大的误差，这种误差随着主元选取的变化有较大的变化，采用列选主元或者全选主元的方法将较大的元素作为主元，使得每次选取得主元绝对值不会接近0从而减小了误差。

​	不限制计算时保留的位数时，计算所得结果在默认保留位数下三种算法得到的结果一致，但当主元不断减小到一定数量级时，三种算法得到的结果又会出现差异，高斯消元会得到一个误差很大的解。

高斯‐若当算法可用于求逆，也可用于求解线性方程组。如下矩阵B用于测试高斯-若当算法。

![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsAyvnlX.png)

计算得到结果如下

![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsksWr7j.png)

计算AB得到结果

![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsw6xATG.png)

​	计算得到单位矩阵，结果符合预期。

若构建增广矩阵时不是增加单位阵而是增加方程组的右边如下

![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wps4jaMF3.png)

计算得到结果如下

![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsuhu0rq.png)

**b**就是矩阵的解，结果保留4位后与理论解相同。

 

**4****.2计算时间对比**

​	三种算法中高斯消元计算过程最简单，列选主元在高斯消元的基础上多了一步行交换，而全选主元又在列选主元的基础上多了列交换及解的位置交换。三种算法的时间复杂度计算得到都是![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsE6XheN.jpg)。三种算法实际运行用时如下

![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsKjtB09.jpg) 

​	由上图可见，程序实际运行所用时间与分析相吻合，全选主元较其余两种算法运行所需时间更多，列选主元及高斯消元用时几乎相同。三种算法都符合时间复杂度为![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpscybXMw.jpg)。

高斯-若当算法时间复杂度也是![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsgcLkzT.jpg)，如下图是高斯-若当求逆算法计算n从10到100所需时间，程序未加速。实际时间符合算法的时间复杂度。

![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsOHCKlg.jpg) 

​	以上用时都是在未加速的情况下获得，若按照未加速时的趋势，5000*5000的矩阵将要花费几天。采用numba对python程序加速后对5000*5000进行消元求解得到下图

![img](https://github.com/ecjtuliwei/GaussianElimination/blob/master/ksohtml/wpsYU8c8C.jpg) 

​	计算用时基本符合预期，高斯-若当求解时要将系数矩阵化为单位阵，用时比其余多，全选主元多了列变换及解的位置变换，用时比列选主元长，列选主元和高斯消元用时基本一样。

​	

## **5** **结论**

​	综合计算结果与原理可得

​	1. 高斯消元效率最高，计算相同大小的矩阵高斯消元法用时最短，在numba装饰器的加速下对5000*5000的矩阵进行消元求解只需一分钟。但高斯消元对系数矩阵有要求，首先是系数矩阵不能是奇异矩阵，其次是对角线元素不能是0，如果出现0作为主元，则高斯消元无法继续进行，当主元不是0而是绝对值很小的数时，由于舍入误差会使结果有较大误差

​	2. 列选主元的高斯消元法在高斯消元的基础上采用选取每一列最大的元素作为主元，通过行变换将最大的元素设为主元进行消元，这样能尽可能避免主元出现0或者绝对值较小的值。但由于每次选主元都是从未消元的行开始选择，随着消元的进行，可以选择的主元越来越少，如果消元到最后剩下的是0，则消元又无法进行了。可供选择的主元在逐渐减少且只进行了行变换，回代求解时解的顺序没有变化，计算量上只增加了寻找最大值及行变换这两步，实际测试发现程序用时与高斯消元法相差不大。

​	3. 列选主元的高斯消元法在最大程度上避免了主元为0及主元绝对值小的情况，在保留位数限制的情况下计算的到的结果最接近理论值。但全选主元对矩阵进行了列变换，列变换使得回代时未知数的位置变换，需要在进行列变换的同时将未知量的顺序同时进行变换，增加了较多的计算量，实际用时接近列选主元的两倍。

​	4. 高斯-若当算法如果用于求解方程组，则不需要回代这一步，但对每一行消元时都要将其他行消元，且多了一步归一化的步骤，实际测试用时超过其余三种算法。如果将高斯-若当用于求逆，则用时会比求解多一倍多。

 

## **参考文献**

[1] 李庆扬. 数值分析(第五版)[M]. 北京: 清华大学出版社. 2008.

 
