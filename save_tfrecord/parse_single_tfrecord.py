import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
filename=os.getcwd()
filename=os.path.join(filename,"tfrecord")
filename=os.path.join(filename,"train.tfrecord")

def parse_record(raw_record):
    keys_to_features = {
        "image": tf.io.FixedLenFeature((), tf.string),
        "label_string": tf.io.FixedLenFeature((), tf.int64)
    }
    #使用parse_sigle_example()函数解析读取的样例
    features=tf.io.parse_single_example(raw_record,
                                     keys_to_features)
    #decode_raw()函数用于将字符串解析成图像对应的像素数组
    images=tf.decode_raw(features["image"],tf.uint8)
    images = tf.reshape(images, [128, 64, 3])
    labels=tf.cast(features["label_string"],tf.int32)
    return images,labels
# 读取所有tfrecord文件得到dataset
dataset = tf.data.TFRecordDataset([filename])
print(dataset)
# 对dataset中的每条数据, 应用parse_record函数, 得到解析后的新的dataset
dataset = dataset.map(parse_record)
dataset = dataset.batch(32)
# 每次sess.run(images, labels)得到一个batch_size的images和labels
iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
images, labels = iterator.get_next()
with tf.Session() as sess:
        print(sess.run(images))