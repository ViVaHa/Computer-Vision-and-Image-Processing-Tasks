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

def constructParameters(length,scale):
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

scales=[1/math.sqrt(2),1,math.sqrt(2),2,2*math.sqrt(2)]
length=7
mat=[[[0 for k in range(length)] for j in range(length)] for i in range(5)]
count=0
for scale in scales:
    temp=constructParameters(7,scale)
    mat[count]=np.asarray(mat[count])
    mat[count]=temp
    count+=1
print(mat[0])