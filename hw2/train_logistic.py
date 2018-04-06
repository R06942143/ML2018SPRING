import numpy as np
import math 
import csv
import sys

def sigmoid(X):
    z = 1.0/(1.0+np.exp(-X))
    return np.clip(z,0.000000001,0.999999999)

def scaling(X):
    a = np.amax(X)
    b = np.amin(X)
    for i in range(len(X)):
        X[i] = (X[i]-b)/(a-b)
    return X




de_feature =[10]

train_x = np.genfromtxt(sys.argv[3],delimiter=',')    #impoert data
train_y = np.genfromtxt(sys.argv[4],delimiter=',')
train_x = np.delete(train_x,0,0)




train_x = np.delete(train_x,de_feature,1)
test_x  = np.genfromtxt(sys.argv[5],delimiter=',')
test_x  = np.delete(test_x,0,0)


test_x = np.delete(test_x,de_feature,1)

for i in range(len(train_x[0])):
    train_x[:,i] = scaling(train_x[:,i])

for i in range(len(test_x[0])):
    test_x[:,i] =  scaling(test_x[:,i])

L = 0.005
epoch = 10000
feature = 122
datasize = len(train_x)
batch_n = 32
batch = int(datasize/batch_n)



w = np.random.rand(feature)
b = np.random.rand(1)
loss_ = np.zeros(feature,dtype = float)

for i in range(epoch):
    print(i,w,b)
    loss = 0
    loss_.fill(0)
    for d in range(batch):
        y = 0
        for k in range(feature):
            y += w[k]*train_x[d][k]
        y = sigmoid(y+b)
        for kk in range(feature):
            loss_[kk] += (1/batch)*(y - train_y[d])*train_x[d][kk]
        loss += (1/batch)*(y - train_y[d])
    
    b -= L*loss
    for k in range(feature):
        w[k] -= L*loss_[k]


predic = np.zeros((len(test_x)))
for k in range(len(test_x)):
    for ww in range(feature):
        predic[k] += test_x[k][ww]*w[ww]
    
    predic[k] = sigmoid(predic[k]+b)
    if(predic[k]>0.5):
            predic[k] = 1
    else:
            predic[k] = 0

ans = []
for i in range(len(predic)):
    ans.append([i+1])
    ans[i].append(int(predic[i]))

filename = str(sys.argv[6])
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
    s.writerow(ans[i])

text.close()