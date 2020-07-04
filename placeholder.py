import tensorflow as tf

# placeholder是tensorflow中的占位符，用来暂时存储变量
# tensorflow 如果想要从外部传入data
# 就需要用到tf.placeholder()  然后以sess.run(***,feed_dict={input: **})

# 在tensorflow中需要定义数据的类型，一般是float32的类型
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

# mul=multiply 是将input1和input2的值计型相乘，并输出为output
output = tf.multiply(input1, input2)

# 传值的工作交给sess.run() 需要传入的值放入feed_dict={} 并且一一对应input ，
# placeholder是与feed_dict{}绑定在一起的
with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1: [7.], input2: [2.]}))
