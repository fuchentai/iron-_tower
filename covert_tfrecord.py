import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import os
import numpy as np


def load_img(file_path):
    """
    file_path: Absolute images file dir
    image_train_num：the number of train dataset
    image_val_num:the nuber of validation dataset
    image_train:data structure-list [np.ndarry,np.ndarry,np.ndarry,..]
    image_val:data structure-list [np.ndarry,np.ndarry,np.ndarry,..]
    label_train:data structure-list
    label_val:data structure-list
    """
    # convertimg(file_path, out_path, width=128, height=256)#在同一目录下，图片转为设定的size
    #file_path图片路径
    file_path_image = os.listdir(file_path)
    label_train=[]
    label_val=[]
    image_train=[]
    image_val=[]
    image_train_num=0
    image_val_num=0
    all_images={}
    label_col=set()#统计类别
    for filename in file_path_image:
        all_label=filename.split("_")[0]
        label_col.add(all_label)
        all_images.setdefault(all_label, [])
        all_images[all_label].append(filename)
    for lab,imgdir in all_images.items():
        number=len(imgdir)
        if number<=50:
            for idx in range(number):
                filename=os.path.join(file_path,imgdir[idx])#路径
                imagefile=os.path.basename(filename)#图片文件名
                label=imagefile.split("_")[0]
                if label=="'":
                    continue
                else:
                    label_train.append(label)
                    img=Image.open(filename)
                    image_arr = np.array(img)
                    image_train.append(image_arr)
        else:
            mask = np.arange(number)
            np.random.shuffle(mask)#打乱数据
            train_idx = mask[:int(number*0.92)]
            val_idx = mask[int(number*0.92):]
            for idx1 in train_idx:
                filename=os.path.join(file_path,imgdir[idx1])
                imagefile=os.path.basename(filename)#图片文件名
                label=imagefile.split("_")[0]
                label_train.append(label)
                img=Image.open(filename)
                image_arr = np.array(img)
                image_train.append(image_arr)
            for idx2 in val_idx:
                filename=os.path.join(file_path,imgdir[idx2])
                imagefile=os.path.basename(filename)#图片文件名
                label=imagefile.split("_")[0]
                label_val.append(label)
                img=Image.open(filename)
                image_arr = np.array(img)
                image_val.append(image_arr)
    image_val_num=len(image_val)
    image_train_num=len(image_train)
    #print(image_train_num)
    return image_train_num,image_val_num,image_train,image_val,label_train,label_val
image_train_num,image_val_num,image_train,image_val,label_train,label_val=load_img(r"E:\研二\20.19.10.1计算机视觉入门\铁塔项目标注\铁塔项目方案\数据集2020331\data_cropped_resize")
#定义生成整数型和字符串型属性的方法，这是将数据填入到Eample协议内存块(Protocol Buffer)的第一步，以后会调用这个方法



def Int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def Bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
#输出TFRecord文件的地址
filename=os.getcwd()
filename=os.path.join(filename,"tfrecord")
filename=os.path.join(filename,"train.tfrecord")
print(filename)
all_labels=['%', '&', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'x', 'y', 'z']
#创建一个Python_io.TFRecordWriter()类的实例
writer=tf.io.TFRecordWriter(filename)
#for循环执行了将数据填入到example协议内存块的主要操作
for i in range(image_train_num):
    #将图像矩阵转换成一个字符串
    image_to_string=image_train[i].astype(np.uint8).tostring()
    label_index=all_labels.index(label_train[i])
    feature={
        "image":Bytes_feature(image_to_string),
        "label_string": Int64_feature(label_index)
    }

    features=tf.train.Features(feature=feature)
    #定义一个Example，将相关的信息写入到这个数据结构
    example=tf.train.Example(features=features)
    #将一个Example写入到TFRecord文件
    #原型writer(self,record)
    writer.write(example.SerializeToString())
#在写完文件后最好的习惯是调用close()函数关闭
writer.close()