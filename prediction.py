import tensorflow as tf
import numpy as np

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# create tf structure
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
print(Weights)
biases = tf.Variable(tf.zeros([1]))
# tf.Variable()用来创建参数


y = Weights * x_data + biases  # 神经网络试着对参数进行优化，学习着将Weights变成0.1，将biases变成0.3

loss = tf.reduce_mean(tf.square(y - y_data))  # 计算电脑通过参数学习的值同真实值之间的误差

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)  # 反向传递误差，使用梯度下降法  利用optimizer来优化这个误差
# 其中0.5表示学习率  minimize() 这个函数完成了梯度计算和参数更新两个工作

init = tf.global_variables_initializer()  # 在使用神经网络之前必须初始化所有的变量  在新版中已经改为tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))
