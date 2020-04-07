import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import os

basefile=os.getcwd()
train_data=pd.read_csv(os.path.join(basefile,'train_data.csv'), encoding='utf-8')
train_label=pd.read_csv(os.path.join(basefile,'train_label.csv'), encoding='utf-8')
test_data=pd.read_csv(os.path.join(basefile,'val_data.csv'), encoding='utf-8')
test_label=pd.read_csv(os.path.join(basefile,'val_label.csv'), encoding='utf-8')

#normalize
train_data = np.array(train_data)
train_data /= 255.0
train_data = train_data.astype(float)
train_data=train_data[:,1:]

val_data = np.array(test_data)
val_data /= 255.0
val_data = test_data.astype(float)
val_data = val_data[:,1:]

print(val_data.shape)
plt.imshow(train_data[10,:].reshape((128,64,3)))
plt.show()

all_labels=['%', '&', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'x', 'y', 'z']
def _change_one_hot_label(X):
    T=np.zeros((X.shape[0],36))
    for idx,row in enumerate(T):
        row[all_labels.index(X[idx,1])]=1
    return T

tr_label=_change_one_hot_label(train_label)
te_label=_change_one_hot_label(test_label)
batch_size = 100
learning_rate = 0.01
max_steps = 30000

def hidden_layer(input_tensor,regularizer,avg_class,resuse):
    #创建第一个卷积层，得到特征图大小为32@28x28
    with tf.variable_scope("C1-conv",reuse=resuse):
        conv1_weights = tf.get_variable("weight", [3, 3, 3, 32],
                             initializer=tf.random_normal_initializer(mean=0.0,stddev=1.0,dtype=tf.float32))
        conv1_biases = tf.get_variable("bias", [32], initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding="SAME")
        relu1 = tf.nn.relu(tf.layers.batch_normalization(tf.nn.bias_add(conv1, conv1_biases)))
    #创建第一个池化层，池化后的结果为32@32*16
    with tf.name_scope("S2-max_pool",):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    # 创建第二个卷积层，得到特征图大小为64@14x14。注意，第一个池化层之后得到了32个
    # 特征图，所以这里设输入的深度为32，我们在这一层选择的卷积核数量为64，所以输出
    # 的深度是64，也就是说有64个特征图
    with tf.variable_scope("C3-conv",reuse=resuse):
        conv2_weights = tf.get_variable("weight", [3, 3, 32, 64],
                                     initializer=tf.random_normal_initializer(mean=0.0,stddev=1.0,dtype=tf.float32))
        conv2_biases = tf.get_variable("bias", [64], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding="SAME")
        relu2 = tf.nn.relu(tf.layers.batch_normalization(tf.nn.bias_add(conv2, conv2_biases)))
    #创建第二个池化层，池化后结果为64@8*16
    with tf.name_scope("S4-max_pool",):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        #get_shape()函数可以得到这一层维度信息，由于每一层网络的输入输出都是一个batch的矩阵，
        #所以通过get_shape()函数得到的维度信息会包含这个batch中数据的个数信息
        #shape[1]是长度方向，shape[2]是宽度方向，shape[3]是深度方向
        #shape[0]是一个batch中数据的个数，reshape()函数原型reshape(tensor,shape,name)
        shape = pool2.get_shape().as_list()
        nodes = shape[1] * shape[2] * shape[3]    #nodes=8192
        reshaped = tf.reshape(pool2, [-1, nodes])
    #创建第一个全连层
    with tf.variable_scope("layer5-full1",reuse=resuse):
        Full_connection1_weights = tf.get_variable("weight", [nodes, 512],
                                      initializer=tf.random_normal_initializer(mean=0.0,stddev=1.0,dtype=tf.float32))
        #if regularizer != None:
        tf.add_to_collection("losses", regularizer(Full_connection1_weights))
        Full_connection1_biases = tf.get_variable("bias", [512],
                                                     initializer=tf.constant_initializer(0.0))
        if avg_class ==None:
            Full_1 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(reshaped, Full_connection1_weights) + \
                                                                   Full_connection1_biases))
        else:
            Full_1 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(reshaped, avg_class.average(Full_connection1_weights))
                                                   + avg_class.average(Full_connection1_biases)))
    #创建第二个全连层
    with tf.variable_scope("layer6-full2",reuse=resuse):
        Full_connection2_weights = tf.get_variable("weight", [512, 36],
                                      initializer=tf.random_normal_initializer(mean=0.0,stddev=1.0,dtype=tf.float32))
        #if regularizer != None:
        tf.add_to_collection("losses", regularizer(Full_connection2_weights))
        Full_connection2_biases = tf.get_variable("bias", [36],
                                                   initializer=tf.constant_initializer(0.0))
        if avg_class == None:
            result = tf.matmul(Full_1, Full_connection2_weights) + Full_connection2_biases
        else:
            result = tf.matmul(Full_1, avg_class.average(Full_connection2_weights)) + \
                                                  avg_class.average(Full_connection2_biases)
    return result

x = tf.placeholder(tf.float32, [None ,128,64,3],name="x-input")
y_ = tf.placeholder(tf.float32, [None, 36], name="y-input")
regularizer = tf.contrib.layers.l2_regularizer(0.001)
y = hidden_layer(x,regularizer,avg_class=None,resuse=False)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
cross_entropy_mean = tf.reduce_mean(cross_entropy)
loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)#, global_step=training_step

with tf.control_dependencies([train_step]):#, variables_averages_op
    train_op = tf.no_op(name='train')
crorent_predicition = tf.equal(tf.arg_max(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(crorent_predicition,tf.float32))

saver=tf.train.Saver()
train_acc_list=[]
train_loss=[]
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(max_steps):
        batch_mask = np.random.choice(14314, batch_size,replace=False)
        x_batch = train_data[batch_mask].reshape((-1,128,64,3))
        y_batch = tr_label[batch_mask]

        _, loss_value, train_acc = sess.run([train_op, loss, accuracy],
                                            feed_dict={x: x_batch, y_: y_batch})
        train_acc_list.append(train_acc)
        train_loss.append(loss_value)
        if i % 1000 == 0:
            print("After %d training steps,train accuracy %g%%" % (i, train_acc * 100))
            saver.save(sess,r"E:\研二\20.19.10.1计算机视觉入门\铁塔项目标注\铁塔项目方案\数据集2020331\saver")

plt.plot(train_loss, markevery=2)
plt.show()