import cv2
import numpy as np
import sys
import os
import copy
from interval import Interval
from PIL import Image
import matplotlib.pyplot as plt
from validation_model import ValidationModel

class Recognizer:
    MAX_WIDTH = 3000
    Min_Area = 1000
    YELLO_HUE_INTERVAL = Interval(11, 34)
    GREEN_HUE_INTERVAL = Interval(35, 99)
    BLUE_HUE_INTERVAL = Interval(100, 124)
    LIMIT_HSV_SATURATION = 34
    GREEN_LIMIT_HSV_SATURATION = 10
    LIMIT_HSV_VALUE = 35
    GREEN_LIMIT_HSV_VALUE = 35
    
    def __init__(self):
        self.blur = 3
        self.morphologyr = 4
        self.morphologyc = 19
        
    def start(self, car_pic):
        img = self.prepare_img(car_pic)
        oldimg = img
        car_contours = self.find_contours(img)
        card_image_models = self.find_card_imgs_with_contours(oldimg, car_contours)
        card_image_models = self.adjust_card_imgs_by_color(card_image_models)
#         self.remove_nails(colors, card_image_models)
        card_image_model = self.segment_card_imgs(card_image_models)
        card_texts = None
        card_img = None
        card_color = None
        if card_image_model is None:
            print("not found card imgs")
        else:
            card_texts = []
            card_img = card_image_model.image
            card_color = card_image_model.color
            part_cards = card_image_model.part_cards
            v = ValidationModel()
            v.is_save_temp_data = True
            for i, part_card in enumerate(part_cards):
#                 cv2.imwrite('temp_result_data/test_%s.bmp' % i, part_card)
#                 show_test_img(part_card)
                if np.mean(part_card) < 255 / 5:
                    continue
                part_card_old = part_card
                charactor = v.recognize(part_card)
                if charactor == "1" and i == 0:
                    continue
                if charactor == "1" and i == len(part_cards) - 1:
                    if part_card_old.shape[0] / part_card_old.shape[1] >= 7:
                        continue
                card_texts.append(charactor)
        return card_texts, card_img, card_color
    
    def prepare_img(self, car_pic):
        if type(car_pic) == type(""):
            img = cv2.imdecode(np.fromfile(car_pic, dtype=np.uint8), cv2.IMREAD_COLOR)
        else:
            img = car_pic
        pic_hight, pic_width = img.shape[:2]

        if pic_width > Recognizer.MAX_WIDTH:
            resize_rate = Recognizer.MAX_WIDTH / pic_width
            img = cv2.resize(img, (Recognizer.MAX_WIDTH, int(pic_hight * resize_rate)), interpolation=cv2.INTER_AREA)
        if self.blur > 0:
            img = cv2.GaussianBlur(img, (self.blur, self.blur), 0)
        return img
    
    def find_contours(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((10, 10), np.uint8)
        img_opening = cv2.morphologyEx(gray_img, cv2.MORPH_OPEN, kernel)
        img_opening = cv2.addWeighted(gray_img, 1, img_opening, -1, 0);
#         show_test_img(img_opening)
        
        ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
#         show_test_img(img_thresh)
        
        img_edge = cv2.Canny(img_thresh, 100, 200)
        kernel = np.ones((self.morphologyr, self.morphologyc), np.uint8)
        img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)
        img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)

        image, contours, hierarchy = cv2.findContours(img_edge1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        image2, contours2, hierarchy2 = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#         self.show_test_contours(gray_img, contours2)
        contours.extend(contours2)
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > Recognizer.Min_Area]
#         self.show_test_contours(img, contours, True)

        print('begin filter contours by ratio')
        car_contours = []
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            area_width, area_height = rect[1]
            print('contour rect:%s,%s,%s ' % (rect), end='')
            if area_width < area_height:
                area_width, area_height = area_height, area_width
            wh_ratio = area_width / area_height
            if wh_ratio > 2 and wh_ratio < 5.5:
                car_contours.append(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                print('result:pass')
            else:
                print('result:fail')
#         self.show_test_contours(gray_img, car_contours)
        print('contours len:%s' % len(car_contours))
        return car_contours
    
    def show_test_contours(self, image, contours, each):
#         if each:
#             for i in range(0, len(contours)):  
#                 image2 = image.copy()
#                 x, y, w, h = cv2.boundingRect(contours[i])   
#                 cv2.rectangle(image2, (x, y), (x + w, y + h), (150, 150, 0), 30) 
#                 show_test_img(image2)
#         else:
#             image2 = image.copy()
#             for i in range(0, len(contours)):  
#                 x, y, w, h = cv2.boundingRect(contours[i])   
#                 cv2.rectangle(image2, (x, y), (x + w, y + h), (150, 150, 0), 30) 
#             show_test_img(image2)
         
        if each:
            for i in range(0, len(contours)):  
                image2 = image.copy()
                cv2.drawContours(image2, contours[i:i + 1], -1, (0, 150, 0), 10)
                show_test_img(image2)
        else:
            image2 = image.copy()
            cv2.drawContours(image2, contours, -1, (0, 150, 0), 10)
            show_test_img(image2)

    
    def find_card_imgs_with_contours(self, img, car_contours):
        pic_hight, pic_width = img.shape[:2]
        card_image_models = []
        for cnt in car_contours:
            rect = cv2.minAreaRect(cnt)
            if rect[2] > -1 and rect[2] < 1:  
                angle = 1
            else:
                angle = rect[2]
            rect = (rect[0], (rect[1][0] + 5, rect[1][1] + 5), angle)  

            box = cv2.boxPoints(rect)
            heigth_point = right_point = [0, 0]
            left_point = low_point = [pic_width, pic_hight]
            for point in box:
                if left_point[0] > point[0] or (left_point[0] == point[0] and left_point[1] > point[1]):
                    left_point = point
                if low_point[1] > point[1] or (low_point[1] == point[1] and low_point[0] > point[0]):
                    low_point = point
                if heigth_point[1] < point[1] or (heigth_point[1] == point[1] and heigth_point[0] < point[0]):
                    heigth_point = point
                if right_point[0] < point[0] or (right_point[0] == point[0] and right_point[1] < point[1]):
                    right_point = point

            if low_point is left_point and heigth_point is right_point:
                card_img = img[int(low_point[1]):int(heigth_point[1]), int(left_point[0]):int(right_point[0])]
                card_image_models.append(CardImageModel(card_img, cnt))
                continue
            if left_point[1] <= right_point[1]: 
                new_right_point = [right_point[0], heigth_point[1]]
                pts2 = np.float32([left_point, heigth_point, new_right_point])
                pts1 = np.float32([left_point, heigth_point, right_point])
                m = cv2.getAffineTransform(pts1, pts2)
                dst = cv2.warpAffine(img, m, (pic_width, pic_hight))
                self.point_limit(new_right_point)
                self.point_limit(heigth_point)
                self.point_limit(left_point)
                card_img = dst[int(left_point[1]):int(heigth_point[1]), int(left_point[0]):int(new_right_point[0])]
                card_image_models.append(CardImageModel(card_img, cnt))
            elif left_point[1] > right_point[1]: 
                new_left_point = [left_point[0], heigth_point[1]]
                pts2 = np.float32([new_left_point, heigth_point, right_point])  
                pts1 = np.float32([left_point, heigth_point, right_point])
                m = cv2.getAffineTransform(pts1, pts2)
                dst = cv2.warpAffine(img, m, (pic_width, pic_hight))
                self.point_limit(right_point)
                self.point_limit(heigth_point)
                self.point_limit(new_left_point)
                card_img = dst[int(right_point[1]):int(heigth_point[1]), int(new_left_point[0]):int(right_point[0])]
                card_image_models.append(CardImageModel(card_img, cnt))
        return card_image_models
    
    def adjust_card_imgs_by_color(self, card_image_models):
        new_card_image_models = []
        for card_index, card_img_model in enumerate(card_image_models):
            card_img = card_img_model.image
#             show_test_img(card_img)
            green = yello = blue = 0
            card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
           
            if card_img_hsv is None:
                continue
            row_num, col_num = card_img_hsv.shape[:2]
            card_img_count = row_num * col_num

            for i in range(row_num):
                for j in range(col_num):
                    H = card_img_hsv.item(i, j, 0)
                    S = card_img_hsv.item(i, j, 1)
                    V = card_img_hsv.item(i, j, 2)
                    if H in Recognizer.YELLO_HUE_INTERVAL and S > Recognizer.LIMIT_HSV_SATURATION:  
                        yello += 1
                    elif H in Recognizer.GREEN_HUE_INTERVAL and S > Recognizer.GREEN_LIMIT_HSV_SATURATION:  
                        green += 1
                    elif H in Recognizer.BLUE_HUE_INTERVAL and S > Recognizer.LIMIT_HSV_SATURATION:  
                        blue += 1
            color = "no"

            def build_new_card_image_model (hue_interval, limit_saturation, limit_value, color):
                new_card_image_model = copy.copy(card_img_model)
                new_card_img = card_img
                xl, xr, yh, yl = self.accurate_place_by_color(card_img_hsv, hue_interval, limit_saturation, limit_value, color)
                if yl != yh or xl != xr:
                    need_retry = False
                    if yl >= yh:
                        yl = 0
                        yh = row_num
                        need_retry = True
                    if xl >= xr:
                        xl = 0
                        xr = col_num
                        need_retry = True
                    new_card_img = card_img[yl:yh, xl:xr] if color != "green" or yl < (yh - yl) // 4 else card_img[yl - (yh - yl) // 4:yh, xl:xr]
                new_card_image_model.image = new_card_img
                new_card_image_model.color = color
                return new_card_image_model
            
            new_card_image_model = None
            if yello * 2.5 >= card_img_count:
                color = "yello"
                new_card_image_model = build_new_card_image_model(Recognizer.YELLO_HUE_INTERVAL, Recognizer.LIMIT_HSV_SATURATION, Recognizer.LIMIT_HSV_VALUE, color)
                new_card_image_models.append(new_card_image_model)
            if green * 2.5 >= card_img_count:
                color = "green"
                new_card_image_model = build_new_card_image_model(Recognizer.GREEN_HUE_INTERVAL, Recognizer.GREEN_LIMIT_HSV_SATURATION, Recognizer.GREEN_LIMIT_HSV_VALUE, color)
                new_card_image_models.append(new_card_image_model)
            if blue * 2.5 >= card_img_count:
                color = "blue"
                new_card_image_model = build_new_card_image_model(Recognizer.BLUE_HUE_INTERVAL, Recognizer.LIMIT_HSV_SATURATION, Recognizer.LIMIT_HSV_VALUE, color)
                new_card_image_models.append(new_card_image_model)
            print('color:%s blue:%s green:%s yello:%s img_count:%s' % (color, blue, green, yello, card_img_count))
        return new_card_image_models
    
    def remove_nails(self, colors, card_image_models):
        for card_index, card_image_model in enumerate(card_image_models):
            card_img = card_image_model.image
            gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
            circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1, 50, param1=30, param2=10, minRadius=2, maxRadius=20)
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cv2.circle(card_img, (i[0], i[1]), i[2], colors[card_index], -1)
            show_test_img(card_img)
    
    def segment_card_imgs(self, card_image_models):
        result = None
        for i, card_image_model in enumerate(card_image_models):
            color = card_image_model.color
            card_img = card_image_model.image
#             show_test_img(card_img)
            gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
            if color == "green" or color == "yello":
                gray_img = cv2.bitwise_not(gray_img)
            ret, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            gray_img = self.trim_gray(gray_img)
#             show_test_img(gray_img)
            x_histogram = np.sum(gray_img, axis=1)
            x_min = np.min(x_histogram)
            x_average = np.sum(x_histogram) / x_histogram.shape[0]
            x_threshold = (x_min + x_average) / 2
            wave_peaks = self.find_waves(x_threshold, x_histogram)
            if len(wave_peaks) == 0:
                print("peak x less 0:")
                continue
            wave = max(wave_peaks, key=lambda x:x[1] - x[0])
#                 show_test_img(gray_img)
            if x_min / x_average < 0.2:
                gray_img = gray_img[wave[0]:wave[1]]
            y_histogram = np.sum(gray_img, axis=0)
            y_min = np.min(y_histogram)
            y_average = np.sum(y_histogram) / y_histogram.shape[0]
            y_threshold = (y_min + y_average) / 5  

            wave_peaks = self.find_waves(y_threshold, y_histogram)
            
            if len(wave_peaks) <= 6:
                print("peak y less 1:", len(wave_peaks))
                continue
            
            wave = max(wave_peaks, key=lambda x:x[1] - x[0])
            max_wave_dis = wave[1] - wave[0]
            if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis / 3 and wave_peaks[0][0] == 0:
                wave_peaks.pop(0)
            
            cur_dis = 0
            for i, wave in enumerate(wave_peaks):
                if wave[1] - wave[0] + cur_dis > max_wave_dis * 0.6:
                    break
                else:
                    cur_dis += wave[1] - wave[0]
            if i > 0:
                wave = (wave_peaks[0][0], wave_peaks[i][1])
                wave_peaks = wave_peaks[i + 1:]
                wave_peaks.insert(0, wave)
            
            point = wave_peaks[2]
            if point[1] - point[0] < max_wave_dis / 3:
                point_img = gray_img[:, point[0]:point[1]]
                if np.mean(point_img) < 255 / 5:
                    wave_peaks.pop(2)
            
            if len(wave_peaks) <= 6:
                print("peak y less 2:", len(wave_peaks))
                continue
            tmp_part_cards = []
            for wave in wave_peaks:
                tmp_part_cards.append(gray_img[:, wave[0]:wave[1]])
            part_cards = []
            for i, part_card in enumerate(tmp_part_cards):
                if np.mean(part_card) < 255 / 5:
                    continue
                part_cards.append(part_card)
            if len(part_cards) <= 6:
                print("part_cards len less:", len(part_cards))
                continue
            card_image_model.part_cards = part_cards
            result = card_image_model
            break
        return result
    
    def trim_gray(self, gray_img):
        row_num, col_num = gray_img.shape[:2]
        xl = 0
        xr = col_num
        yl = 0
        yh = row_num
        for i in range(row_num):
            all_white = True
            all_black = True
            for j in range(col_num):
                c = gray_img[i, j]
                all_white = all_white and c == 255
                all_black = all_black and c == 0
            if not all_white and not all_black:
                break
            yl += 1
        for i in range(row_num - 1, 0, -1):
            all_white = True
            all_black = True
            for j in range(col_num):
                c = gray_img[i, j]
                all_white = all_white and c == 255
                all_black = all_black and c == 0
            if not all_white and not all_black:
                break
            yh -= 1
        for j in range(col_num):
            all_white = True
            all_black = True
            for i in range(row_num):
                c = gray_img[i, j]
                all_white = all_white and c == 255
                all_black = all_black and c == 0
            if not all_white and not all_black:
                break
            xl += 1
        for j in range(col_num - 1, 0, -1):
            all_white = True
            all_black = True
            for i in range(row_num):
                c = gray_img[i, j]
                all_white = all_white and c == 255
                all_black = all_black and c == 0
            if not all_white and not all_black:
                break
            xr -= 1
        if xl < xr and yl < yh:
            gray_img = gray_img[yl:yh, xl:xr]
        return gray_img

    def point_limit(self, point):
        if point[0] < 0:
            point[0] = 0
        if point[1] < 0:
            point[1] = 0

    def find_waves(self, threshold, histogram):
        up_point = -1 
        is_peak = False
        if histogram[0] > threshold:
            up_point = 0
            is_peak = True
        wave_peaks = []
        for i, x in enumerate(histogram):
            if is_peak and x < threshold:
                if i - up_point > 2:
                    is_peak = False
                    wave_peaks.append((up_point, i))
            elif not is_peak and x >= threshold:
                is_peak = True
                up_point = i
        if is_peak and up_point != -1 and i - up_point > 4:
            wave_peaks.append((up_point, i))
        return wave_peaks
    
    def accurate_place_by_color(self, card_img_hsv, hue_interval, limit_saturation, limit_value, color):
        row_num, col_num = card_img_hsv.shape[:2]
        xl = col_num
        xr = 0
        yh = 0
        yl = row_num
        row_num_limit = row_num * 0.3
        col_num_limit = col_num * 0.7 if color != "green" else col_num * 0.5
        row_white_line_limit = row_num * 0.9
        col_white_line_limit = col_num * 0.9
        for i in range(row_num):
            count = 0
            white_count = 0
            for j in range(col_num):
                H = card_img_hsv.item(i, j, 0)
                S = card_img_hsv.item(i, j, 1)
                V = card_img_hsv.item(i, j, 2)
                if H in hue_interval and limit_saturation < S and limit_value < V:
                    count += 1
                if 0 < H < 180 and 0 < S < 43 and 221 < V < 255:
                    white_count += 1
            if count > col_num_limit:
                if yl > i:
                    yl = i
                if yh < i:
                    yh = i
            if white_count > col_white_line_limit:
                if yl >= i:
                    yl = i + 1
                if yh <= i:
                    yh = i - 1
        for j in range(col_num):
            count = 0
            white_count = 0
            for i in range(row_num):
                H = card_img_hsv.item(i, j, 0)
                S = card_img_hsv.item(i, j, 1)
                V = card_img_hsv.item(i, j, 2)
                if H in hue_interval and limit_saturation < S and limit_value < V:
                    count += 1
                if 0 < H < 180 and 0 < S < 43 and 221 < V < 255:
                    white_count += 1
            if count > row_num_limit:
                if xl > j:
                    xl = j
                if xr < j:
                    xr = j
            if white_count > row_white_line_limit:
                if xl > j:
                    xl = j + 1
                if xr < j:
                    xr = j - 1
        return xl, xr, yh, yl
    
class CardImageModel:
    __slots__ = ('image', 'contour', 'color', 'part_cards')
    def __init__(self, image, contour):
        self.image = image
        self.contour = contour
    
def show_test_img(img):
#     cv2.namedWindow("b", 0);
#     cv2.imshow('b', img)
#     cv2.waitKey(0)
    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    c = Recognizer()
    card_result, card_img_model, card_color = c.start("test_image/29.jpg")
    print(card_result)
