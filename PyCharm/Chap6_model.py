#!/usr/bin/env python3
# -*- coding: utf-8 -*-

' test module '

__author__ = 'Liangjun Jia'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np

BATCH_SIZE = 100
LAERNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.99
TRAINING_STEPS = 1000
MOVING_AVERAGE_DECAY = 0.99
# 描述模型复杂度的正则化项在损失函数中的系数
REGULARIZATION_RATE = 0.0001

MODEL_SAVE_PATH = 'D:/PythonCode/Machine/jupyter1/Tensor/path/to/MNIST_data'

INPUT_NODE=784
OUTPUT_NODE=10

IMAGE_SIZE=28
NUM_CHANNELS=1
NUM_LABELS=10

CONV1_DEEP=32
CONV1_SIZE=5

CONV2_DEEP=64
CONV2_SIZE=5

FC_SIZE=512

#参数train用于区分训练过程还是测试过程
def inference(input_tensor,train,regularizer):
    # 声明第一层卷积层，输入28*28*1，输出28*28*32
    with tf.variable_scope('layer1_conv1'):
        conv1_Weight=tf.get_variable('Weight',[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biase=tf.get_variable('biase',[CONV1_DEEP],initializer=tf.constant_initializer(0.0))
        conv1=tf.nn.conv2d(input_tensor,conv1_Weight,strides=[1,1,1,1],padding='SAME')
        relu1=tf.nn.relu(tf.nn.bias_add(conv1,conv1_biase))

    # 声明第一层池化层，输入28*28*32，输出14*14*32
    with tf.variable_scope('layer2_pool1'):
        pool1=tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    # 声明第三层卷积层，输入14*14*32，输出14*14*64
    with tf.variable_scope('layer3_conv2'):
        conv2_Weight=tf.get_variable('Weight',[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],
                                     initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biase=tf.get_variable('biase',[CONV2_DEEP],initializer=tf.constant_initializer(0.0))
        conv2=tf.nn.conv2d(pool1,conv2_Weight,strides=[1,1,1,1],padding='SAME')
        relu2=tf.nn.relu(tf.nn.bias_add(conv2,conv2_biase))

    # 声明第四层池化层，输入14*14*64，输出7*7*64
    with tf.variable_scope('layer4_pool2'):
        pool2=tf.nn.max_pool(relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    # 将第四层输出格式（7*7*64)转化为第五层的输入格式一个向
    pool_shape=pool2.get_shape().as_list()
    #pool_shape[0]存储的是一个batch中的数据个数
    nodes=pool_shape[1]*pool_shape[2]*pool_shape[3]
    reshaped=tf.reshape(pool2,[pool_shape[0],nodes])

    # 声明第五层全连接层，输入7*7*64=3136长度的向量，输出512
    with tf.variable_scope('layer5_fc1'):
        fc1_Weight=tf.get_variable('Weight',[nodes,FC_SIZE],
                                   initializer=tf.truncated_normal_initializer(stddev=0.1))
        #只有全连接层的权重需要加入正则化
        if regularizer!=None:
            tf.add_to_collection('losses',regularizer(fc1_Weight))
        fc1_biase=tf.get_variable('biase',[FC_SIZE],initializer=tf.constant_initializer(0.1))
        fc1=tf.nn.relu(tf.matmul(reshaped,fc1_Weight)+fc1_biase)
        if train:
            #用于防止过拟合,dropout随机将部分节点输出改为0，一般只在全连接层使用
            fc1=tf.nn.dropout(fc1,0.5)

    # 声明第6层全连接层，输入512，输出10，通过softmax之后得到最后的分类结果
    with tf.variable_scope('layer6_fc2'):
        fc2_Weight=tf.get_variable('Weight',[FC_SIZE,NUM_LABELS],
                                   initializer=tf.truncated_normal_initializer(stddev=0.1))
        #只有全连接的权重需要加入正则化
        if regularizer!=None:
            tf.add_to_collection('losses',regularizer(fc2_Weight))
        fd2_biase=tf.get_variable('biase',[NUM_LABELS],initializer=tf.constant_initializer(0.1))
        prediction=tf.nn.relu(tf.matmul(fc1,fc2_Weight)+fd2_biase)

    return prediction

def train(mnist):
    # 卷积神经网络的出入层为三层
    X = tf.placeholder(tf.float32, [
        BATCH_SIZE,  # 第一维表示一个batch中样例个数
        IMAGE_SIZE,  # 第二维和第三维表示图像的尺寸
        IMAGE_SIZE,
        NUM_CHANNELS  # 图像的深度
    ], name='x_input')
    y = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y_input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    pre_y = inference(X, False, regularizer)

    global_step = tf.Variable(0, trainable=False)
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_average_op = variable_average.apply(tf.trainable_variables())

    cross_entrogy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pre_y, labels=tf.argmax(y, 1))
    cross_entrogy_mean = tf.reduce_mean(cross_entrogy)
    loss = cross_entrogy_mean + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(LAERNING_RATE_BASE, global_step,
                                               mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variable_average_op]):
        train_op = tf.no_op(name='train')

    # saver=tf.train.Saver()

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(TRAINING_STEPS):
            Xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshape_xs = np.reshape(Xs,
                                    [BATCH_SIZE,
                                     IMAGE_SIZE,
                                     IMAGE_SIZE,
                                     NUM_CHANNELS])
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={X: reshape_xs, y: ys})
            if i % 10 == 0:
                print('After %d training steps, loss on training batch is %g' % (step, loss_value))

                # saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)


mnist = input_data.read_data_sets('D:\PythonCode\jupyter\Machine\Tensor\path/to\MNIST_data', one_hot=True)

train(mnist)
