#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 20:02:50 2018

@author: varshath
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 19:15:53 2018

@author: varshath
"""
import math
import numpy as np
def constructGaussianBlurOperator(x,y,scale):
    '''x*=scale
    y*=scale'''
    denominator=1
    numerator=1
    denominator*=2
    denominator*=3.14
    denominator*-(scale*scale)
    exp=((x*x)+(y*y))
    exp/=(2*(scale*scale))
    exp=-exp
    numerator*=math.exp(exp)
    x=numerator/denominator
    return x

def helper(length,scale):
    X=[[0 for j in range(length)] for i in range(length)]
    xval=0
    yval=0
    l_limit=int((length/2))
    u_limit=int(length/2)
    l_limit=-l_limit
    if length%2==1:
        u_limit+=1
    for i in range(l_limit,u_limit):
        yval=0
        for j in range(l_limit,u_limit):
            t=constructGaussianBlurOperator(i,j,scale)
            X[xval][yval]=t
            yval+=1
        xval+=1
    return np.asarray(X)

def selectScaleForOctave(selected_octave):
    x=math.sqrt(2) 
    octave=[[0 for i in range(5)] for j in range(4)]
    octave[0]=[1/x,1,x,2,2*x]
    octave[1]=[x,2,2*x,4,4*x]
    octave[2]=[2*x,4,4*x,8,8*x]
    octave[3]=[4*x,8,8*x,16,16*x]
    scales=octave[selected_octave-1]
    length=7
    mat=[[[0 for k in range(length)] for j in range(length)] for i in range(5)]
    count=0
    for scale in scales:
        temp=helper(7,scale)
        mat[count]=np.asarray(mat[count])
        mat[count]=temp
        count+=1
    return mat

def scaleImageForEachOctave():
    img = cv2.imread("/Users/varshath/TensorFlow/task2.jpg",0)
    height=img.shape[0]
    width=img.shape[1]
    prev_mat=img
    imgs=[]
    for times in range(4):
        new_mat=[[0 for j in range(int(width/2))] for i in range(int(height/2))]
        row=0
        col=0
        for i in range(height):
            for j in range(width):
                if j%2==0:
                    new_mat[int(i/2)][int(j/2)]=prev_mat[i][j]
        matrix = np.abs(new_mat) / np.max(np.abs(new_mat)) 
        imgs.append(matrix)
        prev_mat=new_mat
        new_mat=None
        height=len(prev_mat)
        width=len(prev_mat[0])
        if height%2==1:
            height-=1
        if width%2==1:
            width-=1

mat=selectScaleForOctave(1)




