# 搭建神经网络基本流程
# 1、训练数据
# 2、定义节点接收数据
# 3、定义神经层：隐藏层和预测层
# 4、定义loss函数
# 5、选择optimizeer使得loss达到最小

# 然后对所有变量进行初始化，通过sess.run  optimizer迭代1000进行学习

import tensorflow as tf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


# 添加层
def add_layer(inputs, in_size, out_size, activation_function=None):
    # 输入值、输入的大小、输出的大小、激励函数  默认情况下激励函数是None
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # 这里的Weights是一个in_size行  out_size列的随机变量矩阵
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    # biases的推荐值不是0  这里是在0向量的基础上加了0.1 tf.zeros()默认构造向量是0向量
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    # 神经网络未激活的值  tf.matmul() 是矩阵的乘法
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs
    # 当激励函数是空时输出未激励的值 激励函数不是空时，将Wx_plus_b传入激励函数中运算


# 1、训练的数据
# make up some real data
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# 2、定义节点接收数据
# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
# 利用占位符定义我们需要的神经网络的输入，None代表无论输入多少都可以 1代表输入只有一个特征

# 3、定义神经层：隐藏层和预测层
# add hidden layer 输入值是xs,在隐藏层有10个神经元
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer  输入值是隐藏层 l1,在预测层输出一个结果
prediction = add_layer(l1, 10, 1, activation_function=None)

# 4、定义loss表达式
# the error between prediction and real data
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
# 均方差

# 5、选择optimizer 使loss最小
# 这一步是非常关键的一步，告知了机器如何提高准确率，tf.train.GradientDescentOptimizer()的值通常小于1
# 这里取0.1 表示以0.1的效率最小化误差loss
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# important step 对所有变量初始化
init = tf.global_variables_initializer()
sess = tf.Session()
# 上面定义的都没有运算，直到sess.run才会开始运算
sess.run(init)

plt.ion()
plt.scatter(x_data,y_data)

# fig=plt.figure() #定义一个图片框
# ax=fig.add_subplot(1,1,1)  # 定义一个1*1的显示框架
# ax.scatter(x_data,y_data)  #输出样本值  散点图
# plt.show() #一次输出，将程序暂停

# 迭代1000次进行学习，sess.run  optimizer
for i in tqdm(range(1001)):
    # training train_step和loss都是由placeholder定义的运算，所以这里用户feed传入参数
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # to see the step improvement:
        # 可视化显示
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        # lines = ax.plot(x_data, prediction_value, 'r-', lw=5) # 每50次刷新一次图形  用红色、宽度为5的线来显示 并暂停0.1s
        plt.scatter(x_data,y_data)
        plt.plot(x_data, prediction_value, 'r-', lw=5) # 每50次刷新一次图形  用红色、宽度为5的线来显示 并暂停0.1s
        # fig.show()

        plt.pause(0.2)
        plt.clf()
        # print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
