# 回归问题是用来预测一个值 如天气、房价等等
# 一个比较常见的回归算法是线性回归算法（LR）
# 回归分析用在神经网络上 最上层是不需要加上softmax函数，而是直接对前一层累加即可
# 回归强调对真实值的一种逼近预测 本实例中用一条线段对一些连续的数据进行拟合 进而预测出新的输出值

import numpy as np

np.random.seed(1337)  # 指定每次生成的随机值都相同  1337为开始生成的整数值

from keras.models import Sequential  # Sequential 序贯模型是多个网络层的线性堆叠，也就是“一条路走到黑”
from keras.layers import Dense  # Dense 表是全连接层
import matplotlib.pyplot as plt  # 图形化显示

# 创建一些数据
X = np.linspace(-1, 1, 200)  # np.linspace(start,end,num)  在start 到 end 之间返回num个数字的等差数列
np.random.shuffle(X)  # 随机化函数   对数组元素进行重新随机排序  只适用于一维数组
Y = 0.5 * X + 2 + np.random.normal(0, 0.05, (200,))  # np.random.normal()正态分配函数    根据线性函数生成Y，并对Y并加入噪声

# 绘制数据
plt.scatter(X, Y)  # 散点图  X表示x轴坐标  Y 表示y轴坐标
plt.show()

X_train, Y_train = X[:160], Y[:160]  # first 160 data
X_test, Y_test = X[160:], Y[160:]  # last 40 data

# 从第一层到最后一层建立神经网络
model = Sequential()  # 创建一个序贯网络模型
model.add(Dense(units=1, input_dim=1))  # input_dim输入数据的维度，units表示神经元数，即输出单元数
# 如果需要添加下一个神经层，则不要再定义输入维度，因为其默认将前一层的输出作为当前层的输入

# 选择损失函数和优化方法
model.compile(loss='mse', optimizer='sgd')  # model.compile() 用来编译和激活模型
# mse均方误差函数，最常用的回归损失函数，表示预测值同目标值之间的差的平方
# sgd表示随机梯度下降函数

# 训练
print('Training:')
for step in range(301):
    # 进行训练 返回损失（代价）函数
    cost = model.train_on_batch(X_train, Y_train)  # 一批一批的训练X_train和Y_train 默认的返回值是cost
    # 每100步输出一次结果
    # train_on_batch 该函数再一个batch的数据上进行一次参数更新，
    # 返回训练误差的标量值或标量值的list,同evaluate情形相同
    if step % 100 == 0:
        print('train cost:', cost)

# test
print('\ntesting:')
cost = model.evaluate(X_test, Y_test, batch_size=40)  # 输入测试集的数据 输出cost
# model.evaluate(self,x,y,batch_size) 按batch计算在某些输入数据上 模型产生的误差
# 参数x：输入数据，是numpy array或numpy array 的list
# 参数y:标签，是numpy array   参数batch_size:整数
print('test cost:', cost)
W, b = model.layers[0].get_weights()  # 输出weights 和 biase 是在模型的第一层model.layers[0] 学习到的参数
print('Weights:', W, '\nbiases=', b)

# 绘制预测图
Y_pred = model.predict(X_test)  # 根据输入数据输出预测值 返回值是预测值的numpy array
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)        # plt.plot(x,y) 根据输入x和y值输出成一条线
plt.show()
