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
import cv2
def constructGaussianBlurOperator(x,y,scale):
    '''x*=scale
    y*=scale'''
    denominator=1
    numerator=1
    denominator*=2
    denominator*=3.14
    denominator*=(scale*scale)
    exp=((x*x)+(y*y))
    exp/=(2*(scale*scale))
    exp=-exp
    numerator*=math.exp(exp)
    x=numerator/denominator
    return x

def helper(length,scale):
    X=[[0 for j in range(length)] for i in range(length)]
    totalSum=0
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
            totalSum+=t
            yval+=1
        xval+=1
    for i in range(l_limit,u_limit):
        for j in range(l_limit,u_limit):
            X[i][j]/=totalSum
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
    image_one = np.abs(prev_mat) / np.max(np.abs(prev_mat)) 
    imgs.append(image_one)
    for times in range(3):
        new_mat=[[0 for j in range(int(width/2))] for i in range(int(height/2))]
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
    return imgs



def blurImage(img,kernel):
    height=len(img)
    width=len(img[0])
    padded_img=[[0 for j in range(width+6)] for i in range(height+6)]
    for i in range(3,height+3):
        for j in range(3,width+3):
            padded_img[i][j]=img[i-3][j-3]
    convoluted=[[0 for j in range(width+6)] for i in range(height+6)]
    for i in range(3,height+3):
        for j in range(3,width+3):
            val=0
            row=0
            col=0
            totalSum=0
            for k in range(i-3,i+4):
                for l in range(j-3,j+4):
                    val=kernel[row][col]*padded_img[k][l]
                    totalSum+=val
                    col+=1
                row+=1
                col=0
            convoluted[i][j]=totalSum
    blurred_image = np.abs(convoluted) / np.max(np.abs(convoluted))
    return blurred_image




def generateBlurredImages():
    blurredImages=[]
    images=scaleImageForEachOctave()
    for i in range(1,len(images)+1):
        kernels=selectScaleForOctave(i)
        for kernel in kernels:
            blurredImage=blurImage(images[i-1],kernel)
            blurredImages.append(blurredImage)
    return blurredImages


def findDiff(matrix1,matrix2):
    diff=matrix1
    for i in range(len(matrix1)):
        for j in range(len(matrix1[0])):
            diff[i][j]=matrix1[i][j]-matrix2[i][j]
    return diff


def generateDOG(blurredImages):
    octaves=[]
    for i in range(0,20,5):
        octaves.append(blurredImages[i:i+5])
    differences=[]
    for octave in octaves:
        l=len(octave)
        i=0
        inner=[]
        while i<l-1:
            ans=findDiff(octave[i],octave[i+1])
            i+=1
            inner.append(ans)
        differences.append(inner)
    return differences



def findMaxAndMin(lowerIndex,centerIndex,upperIndex,diff,maximums,minimums):
    lowerMatrix=diff[lowerIndex]
    centerMatrix=diff[centerIndex]
    upperMatrix=diff[upperIndex]
    for i in range(3,len(centerMatrix)-3):
        for j in range(3,len(centerMatrix[0])-3):
            lower=lowerMatrix[i][j-1:j+2]
            upper=upperMatrix[i][j-1:j+2]
            center=centerMatrix[i][j-1:j+2]
            maxVal=max(max(lower),max(center),max(upper))
            minVal=min(min(lower),min(center),min(upper))
            if centerMatrix[i][j]==maxVal:
                maximums.append((i,j))
            if centerMatrix[i][j]==minVal:
                minimums.append((i,j))


def hasBothUpperAndLower(i,listToExamine):
    if i-1>=0 and i<len(listToExamine)-1:
        return True
    return False    

def findKeyPoints(differences,images):
    maxKeyPoints=[]
    minKeyPoints=[]
    for diff in differences:
        l=len(diff)
        maximums=[]
        minimums=[]
        for i in range(0,l):
            if hasBothUpperAndLower(i,diff):
                findMaxAndMin(i-1,i,i+1,diff,maximums,minimums)
        maxKeyPoints.append(maximums)
        minKeyPoints.append(minimums)
        
        
    new_images=[]
    for i in range(len(maxKeyPoints)):
        new_image=[[0 for k in range(len(images[i][0]))] for l in range(len(images[i]))]
        for j in range(len(maxKeyPoints[i])):
            tup=maxKeyPoints[i][j]
            x,y=tup[0],tup[1]
            new_image[x][y]=1
        new_images.append(new_image)
    
    for i in range(len(minKeyPoints)):
        new_image=new_images[i]
        for j in range(len(minKeyPoints[i])):
            tup=minKeyPoints[i][j]
            x,y=tup[0],tup[1]
            new_image[x][y]=1
        new_images[i]=new_image
    return new_images



blurredImages=generateBlurredImages()
differences=generateDOG(blurredImages)
new_images=findKeyPoints(differences,blurredImages)




