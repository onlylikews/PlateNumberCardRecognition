import sys
import os
import math
import cv2
import hashlib
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import tensorflow as tf
 
from PIL import Image
from PIL import ImageDraw
import matplotlib.image as mpimg
from train_model import TrainModel

class ValidationModel:
    TEMP_DATA_DIR = "temp_result_data/"
    
    def __init__(self):
        self.is_save_temp_data = False
        self.saver = tf.train.import_meta_graph("%smodel.ckpt.meta" % (TrainModel.MODEL_SAVER_DIR))

    def progress_img(self, part_card):
#         show_test_img(img)
        if type(part_card) == type(""):
            img = Image.open(part_card)
            img = img.convert('L')
            img = array(img)
        else:
            img = part_card
        
        origin_height, origin_width = img.shape[:2]
        for i in range(0, origin_height):
            for j in range(0, origin_width):
                if(img[i, j]!=0):
                    a=1
                if img[i, j] < 128:
                    img[i, j] = 0
                else:
                    img[i, j] = 255
                    
        img = self.image_trim(img)
        origin_height, origin_width = img.shape[:2]
                    
        padding = 2
        dst_width = TrainModel.WIDTH
        dst_height = TrainModel.HEIGHT
        
        new_img = Image.new("RGB", (dst_width, dst_height))
        draw = ImageDraw.Draw(new_img)
        draw.rectangle((0, 0, dst_width, dst_height), fill=(0, 0, 0, 255))
        
        resize_width = dst_width - 2 * padding
        resize_height = dst_height - 2 * padding
        scale = resize_height / origin_height
        
        if scale * origin_width < resize_width:
            left = int((resize_width - self.safe_float2int(scale * origin_width)) / 2)
            img = self.imresize(img, (self.safe_float2int(scale * origin_width), resize_height))
            draw.bitmap((left + padding, padding), img)
        else:
            scale = resize_width / origin_width
            top = int((resize_height - self.safe_float2int(scale * origin_height)) / 2)
            img = self.imresize(img, (resize_width, self.safe_float2int(scale * origin_height)))
            draw.bitmap((padding, top + padding), img)
            
        new_img = new_img.convert('L')
        return new_img
    
    def image_trim(self, img):
        row_num, col_num = img.shape[:2]
        xl = 0
        xr = col_num
        yl = 0
        yh = row_num
        for i in range(row_num):
            need_trim = True
            for j in range(col_num):
                if img[i, j] != 0:
                    need_trim = False
                    break
            if need_trim == False:
                break
            yl += 1
        for i in range(row_num - 1, 0, -1):
            need_trim = True
            for j in range(col_num):
                if img[i, j] != 0:
                    need_trim = False
                    break
            if need_trim == False:
                break
            yh -= 1
        for j in range(col_num):
            need_trim = True
            for i in range(row_num):
                if img[i, j] != 0:
                    need_trim = False
                    break
            if need_trim == False:
                break
            xl += 1
        for j in range(col_num - 1, 0, -1):
            need_trim = True
            for i in range(row_num):
                if img[i, j] != 0:
                    need_trim = False
                    break
            if need_trim == False:
                break
            xr -= 1
            
        if xl < xr and yl < yh:
            img = img[yl:yh, xl:xr]
        return img

    def imresize(self, img, size):
        pil_img = Image.fromarray(uint8(img))
        return pil_img.resize(size)

    def safe_float2int(self, f):
        result = math.floor(f)
        if result <= 0:
            result = 1
        return result

    def recognize(self, part_card):
        with tf.Session() as sess:
            model_file = tf.train.latest_checkpoint(TrainModel.MODEL_SAVER_DIR)
            self.saver.restore(sess, model_file)
     
            graph = tf.get_default_graph()
            predict_max_idx = graph.get_tensor_by_name("pred_network:0")
            input_x = graph.get_tensor_by_name('input_x:0')
            keep_prob = graph.get_tensor_by_name('keep_prob:0')
     
            img = self.progress_img(part_card)
            
            img_data = [[0] * TrainModel.SIZE for i in range(1)]
            width = img.size[0]
            height = img.size[1]
            for h in range(0, height):
                for w in range(0, width):
                    if img.getpixel((w, h)) < 190:
                        img_data[0][w + h * width] = 1
                    else:
                        img_data[0][w + h * width] = 0
                        
            index = sess.run(predict_max_idx, feed_dict={input_x: np.array(img_data), keep_prob: 1.0})[0]
            result = TrainModel.LETTERS_DIGITS[index]
            if self.is_save_temp_data:
                name = hashlib.md5(np.array(img)).hexdigest()
                img.save(ValidationModel.TEMP_DATA_DIR + '%s.bmp' % name)
            return result

def show_test_img(img):
#     cv2.namedWindow("b", 0);
#     cv2.imshow('b', img)
#     cv2.waitKey(0)
    plt.imshow(img) 
    plt.show()

if __name__ == '__main__':
    path = "test_image/1509807306_562_6.bmp"
    m = ValidationModel()
    m.is_save_temp_data = True
    result = m.recognize(path)
    print ("字符是: 【%s】" % result)
