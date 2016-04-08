import numpy as np
import cv2
import time
import sys
import PIL
from StringIO import StringIO
import base64
from random import randint

import threading

import classify_bow as classify

def count_coins(img_name):
    img = cv2.imread(img_name)
    r = 1000.0 / img.shape[1]
    dim = (1000, int(img.shape[0] * r))
    img = cv2.resize(img, dim)

    shifted = cv2.pyrMeanShiftFiltering(img, 21, 20, maxLevel=4)

    gray = cv2.cvtColor(shifted,cv2.COLOR_BGR2GRAY)

    gray_blur = cv2.GaussianBlur(gray, (15, 15), 0)
    thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 1)

    #cv2.imshow("thresh", thresh)

    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=4)

    #cv2.imshow("closing", closing)

    cont_img = closing.copy()
    contours, hierarchy = cv2.findContours(cont_img, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)

    intact = img.copy()
    num_coins = 0
    coin_sum = 0

    for i,cnt in enumerate(contours):
        #cv2.drawContours(img, [cnt], 0, (0,i*50,0), 3)

        area = cv2.contourArea(cnt)
        if 400 <= area <= 100000 and len(cnt) >= 5:

            ellipse = cv2.fitEllipse(cnt)
            x,y = map(int, ellipse[0])
            w,h = map(int, ellipse[1])
            x1, y1 = x-w/2, y-h/2
            x2, y2 = x1+w, y1+h
            if 0.5 < w/float(h) < 2:
                cv2.ellipse(img, ellipse, color = (0,255,0), thickness=2)
                #cv2.rectangle(img, (x1-10,y1-10), (x2+10, y2+10), (0,0,255), 1)
                coin = intact[y1-10:y2+10, x1-10:x2+10]
                value = classify.classify(coin)
                cv2.putText(img, classify.c2f[value], (x - 25, y+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                coin_sum += value
                num_coins+=1

            #cv2.imwrite("saved/"+str(randint(1,100000))+".jpg", coin) # to save found

    print "{} coin(s) found".format(num_coins)
    print "Estimated sum: {} EUR".format(coin_sum/100.0)

    cv2.imwrite('received/processed.png', img)

    #cv2.waitKey(0)
    return num_coins, coin_sum/100.0

#count_coins('img/coinz3.jpg')