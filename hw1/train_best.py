import tensorflow as tf
import numpy as np
import math
import csv
import sys
train = np.genfromtxt(sys.argv[1],delimiter=',')    #impoert data
train = np.delete(train,range(3),1)                 #delete rows
train = np.delete(train,0,0)                        #columns

low = 5
high =110
iteration = 20000
datasize = 5652
feather = 9
L = 0.4
w = np.random.rand(feather)
b = np.random.rand(1)
dell_ = []
for i in range(len(train)):
    if(i%18 != 9):
        dell_.append(i)
train = np.delete(train,dell_,0)
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



x = tf.constant(train_x,dtype = tf.float32,shape=(471*12,9))
y = tf.constant(train_y,shape=(471*12,1),dtype=tf.float32)

w = tf.Variable(tf.random_normal([9,1],dtype=tf.float32))
b = tf.Variable(tf.random_normal([1],dtype=tf.float32))


y_data = tf.add(tf.matmul(x,w),b)

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.AdamOptimizer(L)  #learning rate

train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
#記得初始化
sess.run(init)     #Very important
for step in range(iteration):
    sess.run(train)
    if step % 100 ==0:
        print(step,sess.run(y_data),sess.run(loss))
wwww = np.zeros(9,dtype = float)

wwww = w.eval(session=sess)


dataQ = np.genfromtxt('test.csv',delimiter=',',dtype=float)
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
bb = b.eval(session=sess)

predic = np.zeros((260))
for k in range(260):
    for ww in range(9):
        predic[k] += dataQ[k][ww]*wwww[ww]
    predic[k] += bb

print(predic)
ans = []
for i in range(len(predic)):
    ans.append(["id_"+str(i)])
    ans[i].append(predic[i])

filename = str(sys.argv[2])
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()