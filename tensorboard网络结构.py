import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#载入数据集
mnist = input_data.read_data_sets(u'E:\MNIST_data', one_hot=True)
#每个批次的大小
batch_size = 100
#计算一共有多少个批次,//为整除结果为整数
n_batch = mnist.train.num_examples // batch_size
# 参数概要
def variable_summarizes(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)  # 平均值
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))  # 标准差
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))  # 最大值
        tf.summary.scalar('min', tf.reduce_min(var))  #最小值
        tf.summary.histogram('histogram', var)  # 直方图



with tf.name_scope('input'):
    #定义两个placeholder
    x = tf.placeholder(tf.float32,[None, 784], name='x-input')
    y = tf.placeholder(tf.float32,[None, 10], name='y-input')

with tf.name_scope('layer'):
#创建一个简单的神经网络
    with tf.name_scope('weights'):
        w = tf.Variable(tf.zeros([784,10]), name='w')
        variable_summarizes(w)
    with tf.name_scope('biases'):
        b = tf.Variable(tf.zeros([10]), name='b')
        variable_summarizes(b)
    with tf.name_scope('wx_plus_b'):
        wx_plus_b = tf.matmul(x, w) + b
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(wx_plus_b)

# w2 = tf.Variable(tf.zeros([30,10]))
# b2 = tf.Variable(tf.zeros([10]))
# prediction = tf.nn.softmax(tf.matmul(prediction1, w2) + b2)


#二次代价函数
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.square(y - prediction))
    tf.summary.scalar('loss', loss)
#使用梯度下降法训练优化
with tf.name_scope('train_step'):
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
#初始化变量
init = tf.global_variables_initializer()
#结果存放到一个布尔型变量列表中
with tf.name_scope('correct_prediction'):
    correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(prediction, 1))
#求准确率
with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

# 合并所有summary
merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter(u'E:\MNIST_data\logs', sess.graph)
    for epoch in range(51):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y:batch_ys})
        writer.add_summary(summary, epoch)
        acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels})
        print('Iter' + str(epoch) + 'Testing Accuracy is : ' + str(acc))




