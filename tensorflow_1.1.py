import tensorflow as tf
import numpy as np
#使用numpy生成随机的100个点
x_data = np.random.rand(100)
y_data = x_data*5 + 7

#构造一个线性模型
b = tf.Variable(1.1)
k = tf.Variable(0.5)
y = k*x_data +b

#二次代价函数
loss = tf.reduce_mean(tf.square(y_data - y))
#定义一个梯度下降法作为训练的优化器
optimizer = tf.train.GradientDescentOptimizer(0.2)
train = optimizer.minimize(loss)
#初始变量
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(2000):
        sess.run(train)
        if step % 20 ==0:
            print(step, sess.run([k,b]))
