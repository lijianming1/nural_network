import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#载入数据集
mnist = input_data.read_data_sets(u'E:\MNIST_data', one_hot=True)
#每个批次的大小
batch_size = 100
#计算一共有多少个批次,//为整除结果为整数
n_batch = mnist.train.num_examples // batch_size

#定义两个placeholder
x = tf.placeholder(tf.float32,[None, 784])
y = tf.placeholder(tf.float32,[None, 10])
#创建一个简单的神经网络
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x, w) + b)

# w2 = tf.Variable(tf.zeros([30,10]))
# b2 = tf.Variable(tf.zeros([10]))
# prediction = tf.nn.softmax(tf.matmul(prediction1, w2) + b2)


#二次代价函数
loss = tf.reduce_mean(tf.square(y - prediction))
#使用梯度下降法训练优化
train_step = tf.train.GradientDescentOptimizer(0.8).minimize(loss)
#初始化变量
init = tf.global_variables_initializer()
#结果存放到一个布尔型变量列表中
correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(prediction, 1))
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(100):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y:batch_ys})

        acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
        print('Iter' + str(epoch) + 'Testing Accuracy is : ' + str(acc))




