import numpy as np
import csv
import sys

L = 0.005
epoch = 100
feature = 122

def sigmoid(X):
    z = 1.0/(1.0+np.exp(-X))
    return z

w =   [  1.87831545e+00,-8.91516495e+00,-9.61457431e-01,-1.21899664e+00,
        -5.44896007e-01,-8.73749542e+00,-1.09246111e+00,-7.80655086e-01,
        -7.52742589e-01,-2.80381113e-01,-8.31942439e-01,3.93404346e-03,
        -4.22746301e-01,-5.17471194e-01,1.69796181e+00,-1.75856578e+00,
        -2.74918944e-01,4.66935664e-01,-5.33191967e+00,-8.47458601e-01,
        -3.47497761e-01,-4.55569804e-01,-7.17443705e-01,-9.78262722e-01,
        -6.65453136e-01,-1.68767691e+00,-1.99186277e+00,2.70138383e-01,
        -1.20299363e+00,-1.29085028e+00,-1.82225990e+00,-5.41852266e-02,
        -3.95946264e-01,-1.04970849e+00,1.12879157e+00,9.51976299e-01,
         5.07116258e-01,-5.83469748e-01,-6.05763340e+00,-1.04612076e+00,
        -8.21449161e-01,-3.09446733e-02,1.37912929e+00,-7.13748097e-01,
        -8.12095344e-01,-5.60478747e-01,-6.99286461e-01,1.81783020e+00,
        -1.14284396e+00,1.21159112e+00,1.04924679e+00,-5.56351006e-01,
        -6.92963779e-01,-2.40069434e-01,1.00629449e+00,-4.42587227e-01,
         8.90656859e-02,-2.85129100e-01,3.07000130e-01,4.16477412e-01,
         4.95730340e-01,9.37761486e-01,-2.19094968e+00,7.01415241e-01,
        -1.27232039e+00,-1.60319304e+00,-3.94597560e-01,-9.10912514e-01,
         4.30862576e-01,-5.90718567e-01,-1.55134308e+00,-1.12784123e+00,
        -1.22471547e+00,-1.04482603e+00,-1.64844513e+00,-5.59239149e-01,
         2.82468736e-01,1.64955270e+00,1.10436773e+00,2.91723299e+00,
        -1.21235132e+00,-1.88869572e+00,-2.32055545e+00,-4.82934713e-01,
        -2.62674975e+00,-2.22780943e+00,-1.76629877e+00,-3.04036140e+00,
        -2.41904259e+00,-2.13190079e+00,-3.00461221e+00,-1.29739118e+00,
        -1.41483939e+00,-1.65086651e+00,-1.34839320e+00,-1.32636797e+00,
        -2.16894579e+00,-3.94572806e+00,-1.73932493e+00,-1.38376176e+00,
        -2.40682817e+00,-9.24281693e+00,-2.34409833e+00,-1.98989069e+00,
        -1.54256177e+00,-2.69112992e+00,-2.16255498e+00,-1.77785790e+00,
        -2.05998492e+00,-2.50675297e+00,-1.40606773e+00,-7.27222395e+00,
        -1.04177356e+00,-1.39380431e+00,-2.83506513e+00,-1.91906941e+00,
        -2.38860464e+00,-1.81533957e+00,-9.57538068e-01,-1.97174263e+00,
        -1.77590895e+00,-1.84783041e+00]

b = [ 0.41284913]

de_feature =[10]
train_x = np.genfromtxt(sys.argv[3],delimiter=',')    #impoert data
train_y = np.genfromtxt(sys.argv[4],delimiter=',')
train_x = np.delete(train_x,0,0)


Q0max = np.amax(train_x[:,0])
Q0min = np.amin(train_x[:,0])
for i in range(len(train_x[:,0])):
    train_x[i,0] = (train_x[i,0] -Q0min)/(Q0max -Q0min)

for i in range(len(train_x[:,78])):
    if(train_x[i,78] != 0):
        train_x[i,78] =1
    else:
        train_x[i,78] =0

for i in range(len(train_x[:,79])):
    if(train_x[i,79] != 0):
        train_x[i,79] =1
    else:
        train_x[i,79] =0

Qmax = np.amax(train_x[:,80])
Qmin = np.amin(train_x[:,80])
for i in range(len(train_x[:,80])):
    train_x[i,80] = (train_x[i,80] -Qmin)/(Qmax -Qmin)

train_x = np.delete(train_x,de_feature,1)
test_x  = np.genfromtxt(sys.argv[5],delimiter=',')
test_x  = np.delete(test_x,0,0)
for i in range(len(test_x[:,78])):
    if(test_x[i,78] != 0):
        test_x[i,78] =1
    else:
        test_x[i,78] =0

for i in range(len(test_x[:,79])):
    if(test_x[i,79] != 0):
        test_x[i,79] =1
    else:
        test_x[i,79] =0


for i in range(len(test_x[:,0])):
    test_x[i,0] = (test_x[i,0] -Q0min)/(Q0max -Q0min)

for i in range(len(test_x[:,80])):
    test_x[i,80] = (test_x[i,80] -Qmin)/(Qmax -Qmin)

test_x = np.delete(test_x,de_feature,1)

datasize = len(train_x)


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