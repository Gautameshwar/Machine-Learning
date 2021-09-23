# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 15:41:34 2021

@author: Lenovo
"""

import pandas as pd
import numpy as np

def activate(w,instance):
    s=0
    for i in range(instance.size):
        s+=w[i]*instance[i]
    
    s+=b
    fire=s/abs(s)
    
    return fire

#fn to correct if an instance screws up
def correct(w,instance,label):
        fire=activate(w,instance)
        global b
        #hyperparameter (not so useful in here:(
        epochs=3
        
        error=1/2*(fire-label)
        
        #a misclassification!!
        if fire*label==-1:
            for j in range(epochs):
                
                #how many times you want to perform epochs
                for i in range(w.size):
                    #if error is positive, increase the wt. else decrease it!
                    w[i]*=1+error*instance[i]
                #do we need this here?
                b+=1
                
            return w
        
        else:
            return w

#dataset implemented: https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data

#34 feature columns with real values, 35th column labelling it either good(g) or bad(b)
data=pd.read_csv('Downloads\ionosphere.data')

#each row in inputs is a dataset with features
inputs=np.array([data.iloc[i,0:33].values for i in range(350)])

#ith element==label of the ith row in inputs
labels=data.iloc[:,34].values

#converting binary labels to +-1
labels[labels=='b']=-1
labels[labels=='g']=1

#our weight and bias 
w=np.random.rand((inputs[0].size))
b=1 #do we need this?

#training our weights
for i in range(labels.size):
    w=correct(w,inputs[i],labels[i])


#collecting our predictions
prediction=np.zeros(labels.size)
for i in range(labels.size):
    prediction[i]=activate(w,inputs[i])

#checking our correct predictions
right=0
for i in range(labels.size):
    if prediction[i]==labels[i]:
        right+=1

print('Correct predictions:',right,'/',labels.size)
print("accuracy=",right/labels.size)

'''
Output:
    Correct predictions: 224 / 350
    accuracy= 0.64
'''
