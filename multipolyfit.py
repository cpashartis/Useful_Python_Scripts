# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 14:47:34 2013

@author: christoforos

Quick routine for mutlidimensional least squares fit.

"""

import numpy as np

def multipolyfit(datafile, deg, skiprows = 0, crossterms = False, correlate = False, sigma = 0):
    """Works kinda like this
    
    Fit a line, y = mx + c, through some noisy data-points:

>>> x = np.array([0, 1, 2, 3])
>>> y = np.array([-1, 0.2, 0.9, 2.1])
By examining the coefficients, we see that the line should have a gradient of roughly 1 and cut the y-axis at, more or less, -1.

We can rewrite the line equation as y = Ap, where A = [[x 1]] and p = [[m], [c]]. Now use lstsq to solve for p:

>>> A = np.vstack([x, np.ones(len(x))]).T
>>> A
array([[ 0.,  1.],
       [ 1.,  1.],
       [ 2.,  1.],
       [ 3.,  1.]])
>>> m, c = np.linalg.lstsq(A, y)[0]
>>> print m, c
1.0 -0.95
Plot the data along with the fitted line:

>>> import matplotlib.pyplot as plt
>>> plt.plot(x, y, 'o', label='Original data', markersize=10)
>>> plt.plot(x, m*x + c, 'r', label='Fitted line')
>>> plt.legend()
>>> plt.show()
"""
    
    #load data
    data = np.loadtxt(datafile, skiprows=skiprows)
    total_col = len(data[0])-1 #-1 for response at the end
    total_rows = len(data)
    
    if sigma == 0:
        sigma = 1
    
    if deg == 1:
        #the coefficient matrix is just x**0 and x**1
        deg_f = total_rows - total_col - 1
        
        A = np.vstack(((data[:,0]),np.ones(total_rows))).T #creates x1 first column x0 second
        for i in range(1,total_col):
            A = np.hstack((A,np.vstack((data[:,i])))) #keeps adding on to A
        
    elif deg == 2:
        #coefficient matrix is x**0 x**1 x**2 with or without cross terms
        
        deg_f = total_rows - total_col*2 - 1
        
        A = np.vstack((data[:,0]**2, data[:,0], np.ones((total_rows)))).T
        
        for i in range(1,total_col):
            A = np.hstack((A,np.vstack((data[:,i]**2, data[:,i])).T))
            
        if crossterms == True:
            print """ Cross terms are at the end, in form (say i,j,k were original parameters)
            (i**1)(j**1), (i**1)(k**1), (j**1)(k**1)"""
            
            counter = 0 #for n choose r
            for i in range(total_col-1):
                for j in range(i+1, total_col):
                    counter +=1
                    A = np.hstack((A, np.vstack((data[:,i]*data[:,j]))))
            deg_f +=counter
        
    else:
        raise Exception('deg must be >=1')

    coef, residual_sum, holder1, holder2 = np.linalg.lstsq(A, data[:,-1])
    
    if sigma == 1:
        sigma = residual_sum/(deg_f) #change to degrees of freedom
        
    chi_sq = residual_sum/sigma**2
    chi_sq_red = chi_sq/deg_f
    R_sq = 1 - sum((data[:-1]-np.average(data[:-1]))**2)/residual_sum
    
    if correlate == True:
        corrcoef = np.corrcoef(data,rowvar= 0)
        return coef, chi_sq, chi_sq_red, R_sq, corrcoef
        
    return coef, chi_sq, chi_sq_red, R_sq
        
