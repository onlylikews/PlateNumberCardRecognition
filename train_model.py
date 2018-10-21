import sys
import os
import time
import random
 
import numpy as np
import tensorflow as tf
 
from PIL import Image

class TrainModel:
    WIDTH = 32
    HEIGHT = 40
    SIZE = WIDTH * HEIGHT
     
    MODEL_SAVER_DIR = "model_saver/"
    TRAIN_DATA_DIR = "train_data/"
    VALIDATION_DATA_DIR = "validation_data/" 
     
    LETTERS_DIGITS = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                      "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
                      "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
                      "W", "X", "Y", "Z", "粤")
    NUM_CLASSES = len(LETTERS_DIGITS)
    
    def conv_layer(self, inputs, W, b, conv_strides, kernel_size, pool_strides, padding):
        L1_conv = tf.nn.conv2d(inputs, W, strides=conv_strides, padding=padding)
        L1_relu = tf.nn.relu(L1_conv + b)
        return tf.nn.max_pool(L1_relu, ksize=kernel_size, strides=pool_strides, padding='SAME')
    
    def full_connect(self, inputs, W, b):
        return tf.nn.relu(tf.matmul(inputs, W) + b)
    
    def calc_data_count(self, dir):
        return count
    
    def init_data(self, rootDir):
        count = 0
        for i in range(0, TrainModel.NUM_CLASSES):
            char = TrainModel.LETTERS_DIGITS[i]
            dir = rootDir + '%s/' % char 
            for rt, dirs, files in os.walk(dir):
                for filename in files:
                    count += 1
                    
        images = np.array([[0] * TrainModel.SIZE for i in range(count)])
        labels = np.array([[0] * TrainModel.NUM_CLASSES for i in range(count)])
     
        index = 0
        for i in range(0, TrainModel.NUM_CLASSES):
            char = TrainModel.LETTERS_DIGITS[i]
            dir = rootDir + '%s/' % char   
            for rt, dirs, files in os.walk(dir):
                for filename in files:
                    filename = dir + filename
                    img = Image.open(filename)
                    width = img.size[0]
                    height = img.size[1]
                    for h in range(0, height):
                        for w in range(0, width):
                            if img.getpixel((w, h)) > 230:
                                images[index][w + h * width] = 0
                            else:
                                images[index][w + h * width] = 1
                    labels[index][i] = 1
                    index += 1
        return count, images, labels
    
    def start_train(self, iterations):
        time_begin = time.time()
        input_count, input_images, input_labels = self.init_data(TrainModel.TRAIN_DATA_DIR)
        val_count, val_images, val_labels = self.init_data(TrainModel.VALIDATION_DATA_DIR)
     
        time_elapsed = time.time() - time_begin
        print("读取训练图像耗费时间：%d秒" % time_elapsed)
        print ("一共读取了 %s 个训练图像" % input_count)
        
        time_begin = time.time()
        x = tf.placeholder(tf.float32, shape=[None, TrainModel.SIZE], name="input_x")
        y_ = tf.placeholder(tf.float32, shape=[None, TrainModel.NUM_CLASSES], name="input_y")
        x_image = tf.reshape(x, [-1, TrainModel.WIDTH, TrainModel.HEIGHT, 1])
        
        with tf.Session() as sess:
            W_conv1 = tf.Variable(tf.truncated_normal([8, 8, 1, 16], stddev=0.1), name="W_conv1")
            b_conv1 = tf.Variable(tf.constant(0.1, shape=[16]), name="b_conv1")
            conv_strides = [1, 1, 1, 1]
            kernel_size = [1, 2, 2, 1]
            pool_strides = [1, 2, 2, 1]
            L1_pool = self.conv_layer(x_image, W_conv1, b_conv1, conv_strides, kernel_size, pool_strides, padding='SAME')
     
            W_conv2 = tf.Variable(tf.truncated_normal([5, 5, 16, 32], stddev=0.1), name="W_conv2")
            b_conv2 = tf.Variable(tf.constant(0.1, shape=[32]), name="b_conv2")
            conv_strides = [1, 1, 1, 1]
            kernel_size = [1, 1, 1, 1]
            pool_strides = [1, 1, 1, 1]
            L2_pool = self.conv_layer(L1_pool, W_conv2, b_conv2, conv_strides, kernel_size, pool_strides, padding='SAME')
     
            W_fc1 = tf.Variable(tf.truncated_normal([16 * 20 * 32, 512], stddev=0.1), name="W_fc1")
            b_fc1 = tf.Variable(tf.constant(0.1, shape=[512]), name="b_fc1")
            h_pool2_flat = tf.reshape(L2_pool, [-1, 16 * 20 * 32])
            h_fc1 = self.full_connect(h_pool2_flat, W_fc1, b_fc1)
     
            keep_prob = tf.placeholder(tf.float32, name="keep_prob")
     
            h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
     
            W_fc2 = tf.Variable(tf.truncated_normal([512, TrainModel.NUM_CLASSES], stddev=0.1), name="W_fc2")
            b_fc2 = tf.Variable(tf.constant(0.1, shape=[TrainModel.NUM_CLASSES]), name="b_fc2")
     
            y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
            train_step = tf.train.AdamOptimizer((1e-4)).minimize(cross_entropy)
     
            correct_prediction = tf.equal(tf.argmax(y_conv, 1, name="pred_network"), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
     
            sess.run(tf.global_variables_initializer())
            
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(TrainModel.MODEL_SAVER_DIR)
            if ckpt:
                model_file = tf.train.latest_checkpoint(TrainModel.MODEL_SAVER_DIR)
                saver.restore(sess, model_file)

            batch_size = 60
            batches_count = int(input_count / batch_size)
            remainder = input_count % batch_size
     
            for it in range(iterations):
                for n in range(batches_count):
                    train_step.run(feed_dict={x: input_images[n * batch_size:(n + 1) * batch_size], y_: input_labels[n * batch_size:(n + 1) * batch_size], keep_prob: 0.5})
                if remainder > 0:
                    start_index = batches_count * batch_size;
                    train_step.run(feed_dict={x: input_images[start_index:input_count - 1], y_: input_labels[start_index:input_count - 1], keep_prob: 0.5})
     
                iterate_accuracy = 0
                if it % 5 == 0:
                    iterate_accuracy = accuracy.eval(feed_dict={x: val_images, y_: val_labels, keep_prob: 1.0})
                    print ('第 %d 次训练迭代: 准确率 %0.5f%%' % (it, iterate_accuracy * 100))
     
            print ('完成训练!')
            time_elapsed = time.time() - time_begin
            print ("训练耗费时间：%d秒" % time_elapsed)
            time_begin = time.time()
     
            if not os.path.exists(TrainModel.MODEL_SAVER_DIR):
                os.makedirs(TrainModel.MODEL_SAVER_DIR)         
            saver_path = saver.save(sess, "%smodel.ckpt" % (TrainModel.MODEL_SAVER_DIR))

if __name__ == '__main__':
    m = TrainModel()
    m.start_train(500)
