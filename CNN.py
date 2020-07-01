import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm

# 加载数据集
mnist = input_data.read_data_sets('D:/Code/MNIST_data', one_hot=True)

xs = tf.placeholder(tf.float32, [None, 784])  # 输入数据占位符
ys = tf.placeholder(tf.float32, [None, 10])  # 输入标签占位符


# 定义四个函数 分别用于初始化权值W、 初始化偏置项b 构建卷积层、构建池化层
# 用于初始化所有的权重 W
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 初始化所有的偏置
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 定义卷积 conv2d是tf中的二维卷积函数，x是图片的所有参数  W是此卷积层的权重，然后定义步长
# strides=[1,1,1,1] strides[0]和strides[3] 的两个1是默认值 中间两个一 代表在x方向运动1步  在y方向运动1步
# padding same表示卷积计算时会在张量周围补0  valid表示不会
# SAME方式抽取后 和原图片大小一样 VALID方式抽取后比原图像要小
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# x就表示一张图片的所有信息

# padd时选择一次一步 这样一来得到的图片尺寸没有变化，而我们需要对图片压缩这样参数减少
# 采用pooling来稀疏化参数 即下采样层
# 在这里使用最大值化
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 构建网络 整个网络包含两个卷积层（激活层、池化层） 一个全连接层、一个dropout层、一个softmax层
# 转换图片格式
x_image = tf.reshape(xs, [-1, 28, 28, 1])  # -1 表示任意个数的样本数  大小变成28*28 通道为1  因为是黑白图片
#为了使之能用于计算，我们使用reshape将其转换为四维的tensor，
# 其中第一维的－1是指我们可以先不指定，第二三维是指图像的大小，第四维对应颜色通道数目，灰度图对应1，rgb图对应3.

# conv1 layer
W_conv1 = weight_variable([5, 5, 1, 32])   # patch 5x5
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 第一个卷积层 outsize=28x28x32  因为采用SAME方式 卷积后大小不变
h_pool1 = max_pool_2x2(h_conv1)  # 第一个池化层    outsize 14x14x32  一次跨两步 相当于 SANE方式相当于图像被缩小一倍
#首先在每个5x5网格中，提取出32张特征图。
# 其中weight_variable中前两维是指网格的大小，
# 第三维的1是指输入通道数目，
# 第四维的32是指输出通道数目（也可以理解为使用的卷积核个数、得到的特征图张数）。
# 每个输出通道都有一个偏置项，因此偏置项个数为32。

# conv2 layer
W_conv2 = weight_variable([5, 5, 32, 64])  # patch 5x5, in channel32, out channel 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # no.2 卷积层  out size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)  # no.2 池化层   out size 7x7x64

# 全连接层
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])  # reshape成向量
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # 第一个全连接层

# dropout层
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# softmax层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))  # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # 精确度计算


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result


# run network
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print(compute_accuracy(
            mnist.test.images[:1000], mnist.test.labels[:1000]))
