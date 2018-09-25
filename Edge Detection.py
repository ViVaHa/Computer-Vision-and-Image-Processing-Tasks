#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 16:00:10 2018

@author: varshath
"""

import cv2
from PIL import Image
import numpy as np
im = Image.open('/Users/varshath/TensorFlow/task1.png')
width, height = im.size
img = cv2.imread("/Users/varshath/TensorFlow/task1.png",0)
#sobel=[[0 for i in range(3)] for j in range(3)]
sobel=np.zeros((3,3))
sobel[0][0]=-1
sobel[1][0]=-2
sobel[2][0]=-1
sobel[0][2]=1
sobel[1][2]=2
sobel[2][2]=1
img_x=np.zeros((600,900))
for x in range(1,height-1):
    for y in range(1,width-1):
        topLeft=img[x-1][y-1]*sobel[0][0]
        topRight=img[x-1][y+1]*sobel[0][2]
        bottomLeft=img[x-1][y+1]*sobel[2][0]
        bottomRight=img[x+1][y+1]*sobel[2][2]
        middleLeft=img[x][y-1]*sobel[1][0]
        middleRight=img[x-1][y-1]*sobel[1][2]
        val=topLeft+topRight+bottomLeft+bottomRight+middleLeft+middleRight
        img_x[x][y]=val

pos_edge_x = (img_x - np.min(img_x)) / (np.max(img_x) - np.min(img_x))
cv2.namedWindow('Sobel edge detection along X axis', cv2.WINDOW_NORMAL)
cv2.imshow('pos_edge_x_dir', pos_edge_x)
cv2.waitKey(0)
cv2.destroyAllWindows()

sobel[0][0]=-1
sobel[0][1]=-2
sobel[0][2]=-1
sobel[2][0]=1
sobel[2][1]=2
sobel[2][2]=1
img_y=np.zeros((600,900))
for x in range(1,height-1):
    for y in range(1,width-1):
        topLeft=img[x-1][y-1]*sobel[0][0]
        topRight=img[x-1][y+1]*sobel[0][2]
        bottomLeft=img[x-1][y+1]*sobel[2][0]
        bottomRight=img[x+1][y+1]*sobel[2][2]
        middleLeft=img[x][y-1]*sobel[1][0]
        middleRight=img[x-1][y-1]*sobel[1][2]
        val=topLeft+topRight+bottomLeft+bottomRight+middleLeft+middleRight
        img_y[x][y]=val
pos_edge_y = (img_y - np.min(img_y)) / (np.max(img_y) - np.min(img_y))
cv2.namedWindow('Sobel edge detection along Y axis', cv2.WINDOW_NORMAL)
cv2.imshow('pos_edge_y_dir', pos_edge_y)
cv2.waitKey(0)
cv2.destroyAllWindows()
