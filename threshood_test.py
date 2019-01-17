import cv2
import numpy as np
frame = cv2.imread('err.jpg')
gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
ret,binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imshow('binary',binary)


def on_trackbar(val):
    arg1 = cv2.getTrackbarPos('arg1','binary')*2+1
    arg2 = cv2.getTrackbarPos('arg2','binary')-20
    print(arg1,arg2)
    #203,-17
    th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,arg1,arg2)
    erode = cv2.erode(th2,np.zeros((5,5),np.uint8),iterations = 1)
    th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,arg1,arg2)
    cv2.imshow('erode',erode)
    cv2.imshow('th2',th2)
    cv2.imshow('th3',th3)

cv2.createTrackbar('arg1', 'binary' , 0, 500, on_trackbar)
cv2.createTrackbar('arg2', 'binary' , 0, 40, on_trackbar)
cv2.setTrackbarPos('arg1', 'binary',101)
cv2.setTrackbarPos('arg2', 'binary',20)


cv2.waitKey(0)