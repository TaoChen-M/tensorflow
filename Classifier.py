import tensorflow as tf
from tqdm import tqdm
# 数据导入
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('D:/Code/MNIST_data', one_hot=True)
# MNIST 库是手写字体库，数据中由55000张图片，每张图片分辨率是28*28
# 所以训练网络输入是28*28=784个像素数据

# 搭建网络
xs = tf.placeholder(tf.float32, [None, 784])  # 28*28
# 每张图片表示一个数字  所以输出是数字0到9 共10类
ys = tf.placeholder(tf.float32, [None, 10])


#  define add_layer
def add_layer(inputs, input_size, out_size, activation_function=None):
    # define Weights
    Weights = tf.Variable(tf.random_normal([input_size, out_size]))
    # define biases
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    # define Wx_plus_b
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

# 调用add_layer 函数 搭建一个简单的训练神经网络 只有输入层和输出层
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)

# define loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
# loss函数是交叉熵函数 交叉熵用来衡量预测值同真实值间相似程度，如果完全相同 他们的交叉熵是0

# train方法 提顿下降法
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train  每次只取100张图片
batch_xs, batch_ys = mnist.train.next_batch(100)
sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})

for i in tqdm(range(10001)):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))
