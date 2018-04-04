import numpy as np
import csv
import math
import sys

def sigmoid(qq):
    z = 1.0/(1.0+math.exp(-qq))
    return z


train_x = np.genfromtxt(sys.argv[3],delimiter=',')    #import data
train_y = np.genfromtxt(sys.argv[4],delimiter=',')
train_x = np.delete(train_x,0,0)

test_x  = np.genfromtxt(sys.argv[5],delimiter=',')
test_x  = np.delete(test_x,0,0)



train_c1 = train_x[np.where(train_y ==1)]
train_c2 = train_x[np.where(train_y ==0)]

m_c1 = np.mean(train_c1,axis = 0)
m_c2 = np.mean(train_c2,axis = 0)

cov_c1 = np.cov(train_c1,rowvar= False)*len(train_c1)/len(train_x)
cov_c2 = np.cov(train_c2,rowvar= False)*len(train_c2)/len(train_x)
cov    = cov_c1 + cov_c2
cov_   = np.linalg.pinv(cov)

w      = np.dot((m_c1 - m_c2),cov_)
x = test_x.T
b = (-0.5)*np.dot(np.dot([m_c1],cov_),m_c1)+(0.5)*np.dot(np.dot([m_c2],cov_),m_c2)+np.log(float(len(train_c1))/len(train_c2))
y = np.dot(w,x)+b

for k in range(len(y)):
    y[k] = sigmoid(y[k])
    if(y[k]>0.5):
            y[k] = 1
    else:
            y[k] = 0

ans = []
for i in range(len(y)):
    ans.append([i+1])
    ans[i].append(int(y[i]))

filename = str(sys.argv[6])
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
    s.writerow(ans[i])
text.close()