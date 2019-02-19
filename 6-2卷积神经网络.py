import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets(u'E:\MNIST_data', one_hot=True)
batch_size = 100
# 计算一共有多少个批次
n_batch = mnist.train.num_examples // batch_size


# 初始化权值
def weight_variable(shape):
    # 生成一个截断的正太分布
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 初始化偏量
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积层
def conv2d(x, w):
    # x input tensor of shape[batch,in_height,in_width,in_channels]
    # w filter/kernel tensor of shape[filter_height, filter_width, in_channels, out_channels]
    # strides[0]=strides[3]=1 strides[1]代表x方向的步长，strides[2]代表y方向上的步长
    # padding: a string from: 'SAME','VALID'
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


# 池化层
def max_pool_2x2(x):
    # ksize[1, x, y, 1]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None,10])

# 改变x的格式为4D的向量[batch, in_height, in_width, in_channels]
# batch = -1 表示为任意长度
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 初始化第一个卷积层的权值和偏量
# 5*5的采样窗口，32个卷积核从一个平面抽取特征
w_conv1 = weight_variable([5, 5, 1, 32])
# 每一个卷积核一个偏量值
b_conv1 = bias_variable([32])

# 把x_image和卷积向量进行卷积，再加上偏置值， 然后应用于relu激活函数
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool = max_pool_2x2(h_conv1)

# 初始化第二个权值和偏置值
# 5*5的采样窗口，64个卷积核从32个平面抽取特征
w_conv2 = weight_variable([5, 5, 32, 64])
# 每一个卷积核一个偏置值
b_conv2 = bias_variable([64])
# 把h_pool 和权值向量进行卷积，再加上偏置值，然后应用relu激活函数
h_conv2 = tf.nn.relu(conv2d(h_pool, w_conv2) + b_conv2)
# 进行池化
h_pool2 = max_pool_2x2(h_conv2)

# 28*28的图片第一次卷积后还是28*28，第一次池化后变为32张14*14的平面
# 第二次卷积后还是14*14， 第二次池化后变成64张7*7的平面

# 初始化第一个全连接层的权值
w_fcl = weight_variable([7*7*64, 500])  # 上一层有7*7*64个神经元， 全连接层有1024个神经元
b_fcl = bias_variable([500]) # 1024个节点

# 把池化层2的输出扁平化为1维
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
# 求第一个全连接层输出
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fcl) + b_fcl)

# keep_prob用来表示神经元的输出概率
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 初始化第二个全连接层的权值和偏置值
w_fcl2 = weight_variable([500, 10])
b_fcl2 = bias_variable([10])

# 计算输出
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,w_fcl2) + b_fcl2)
# 交叉熵代价函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
# 使用AdamOptimizer进行优化
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# 结果存放在一个bool列表中
# argmax() 返回一维张量中最大值所在的位置
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(30):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})

        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        print("Iter" + str(epoch) + ", Test accuracy is : " + str(acc))






