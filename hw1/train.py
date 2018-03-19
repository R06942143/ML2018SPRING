import numpy as np
import sys
import math
import csv

# ============================================parameter=================================================
w = np.random.rand(9) # weight
b = np.random.rand(1)   # bias
L =0.0005 #learning rate
iteration = 1000 # numbers of iteration
low = 10
high = 110
datasize = 5652
# ============================================parameter=================================================

#train.columns   = ['date','place','name','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x']


# ============================================get_data===================================================
train = np.genfromtxt(sys.argv[1],delimiter=',')    #impoert data
train = np.delete(train,range(3),1)                 #delete rows
train = np.delete(train,0,0)                        #columns
dell_ = []
for i in range(len(train)):
    if(i%18 !=9):
        dell_.append(i)
train = np.delete(train,dell_,0)
train = np.nan_to_num(train)                        #rainfall
train = np.absolute(train)                          #-1 -> 1

train_set = []
for i in range(len(train)):
    for j in range(len(train[0])):
        if(train[i][j]<0):
            for k in range(10):
                qq = 0.00000000001
                qq_c = 1
                if(train[i][k-5]>low and train[i][k-5]<high):
                    qq += train[i][k]
                    qq_c+=1
            train_set = np.append(train_set,qq/qq_c)
        elif (train[i][j]<high):
            train_set = np.append(train_set,train[i][j])
        else:
            for k in range(10):
                qq = 0
                qq_c = 0.0000000001
                if(train[i][k-5]>low and train[i][k-5]<high):
                    qq += train[i][k]
                    qq_c+=1
            train_set = np.append(train_set,qq/qq_c)

train_x = np.zeros((471*12,9),dtype = float)
train_y = np.zeros((471*12),dtype = float)
for i in range(12):
    for data in range(471):
        for hours in range(9):
            train_x[i*471+data][hours]  = train_set[hours+data+i*480]
        train_y[i*471+data] = train_set[data+i*480+9]



# ============================================predict=====================================
for epoch in range(iteration):
    print(epoch,w,b)
    loss = 0
    loss_ = np.zeros((9,),dtype= float)
    for data_size in range(datasize):
        train_predict = 0
        for i in range(9):
            train_predict += w[i]*train_x[data_size][i]
        train_predict += b
# ============================================compute loss_=================================
        for j in range(9):
            loss_[j] += (1/(datasize))*(train_predict - train_y[data_size])*train_x[data_size][j]
        
# ===========================================square loss function==================================
        loss += (1/(datasize))*(train_predict - train_y[data_size])
    b -= L*loss/math.sqrt(epoch+1)
    # b -= L*loss/math.sqrt(epoch+1)
    for k in range(9):
        w[k] -= L*loss_[k]/math.sqrt(epoch+1)
        # w[k] -= L*loss_[k]/math.sqrt(epoch+1)


dataQ = np.genfromtxt(sys.argv[2],delimiter=',',dtype=float)
dataQ = np.delete(dataQ,range(2),axis=1)
dataQ = np.nan_to_num(dataQ)
dell_ =[]
for i in range(len(dataQ)):
    if(i%18 !=9):
        dell_.append(i)

dataQ = np.delete(dataQ,dell_,0)
for i in range(len(dataQ)):
    for j in range(len(dataQ[0])):
        if(dataQ[i][j]>high or dataQ[i][j]<low):
            if(j != len(dataQ[0])-1):
                dataQ[i][j] = dataQ[i][j+1]
            else:
                dataQ[i][j] = dataQ[i][j-1]

predic = np.zeros((260))
for k in range(260):
    for ww in range(9):
        predic[k] += dataQ[k][ww]*w[ww]
    predic[k] += b

print(predic)
ans = []
for i in range(len(predic)):
    ans.append(["id_"+str(i)])
    ans[i].append(predic[i])

filename = str(sys.argv[3])
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()