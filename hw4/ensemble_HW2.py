from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import numpy as np
import csv
from sklearn.externals import joblib



de_feature =[10]

train_x = np.genfromtxt(r'G:\My Drive\code\python\ML_HW2\data\train_X',delimiter=',')    #impoert data
train_y = np.genfromtxt(r'G:\My Drive\code\python\ML_HW2\data\train_Y',delimiter=',')
train_x = np.delete(train_x,0,0)

Qmax = np.max(train_x[:,78])
Qmin = np.min(train_x[:,78])
for i in train_x[:,78]:
    i = (i-Qmin)/(Qmax-Qmin)


train_x = np.delete(train_x,de_feature,1)
test_x  = np.genfromtxt(r'G:\My Drive\code\python\ML_HW2\data\test_X',delimiter=',')
test_x  = np.delete(test_x,0,0)
for i in test_x[:,78]:
    i = (i-Qmin)/(Qmax-Qmin)
test_x = np.delete(test_x,de_feature,1)
pca = PCA(n_components=50)
pca.fit_transform(train_x)

clf = joblib.load(r'H:\master\code\python\ML_HW2\model1.pkl') 
test_y1 = clf.predict(test_x)

clf = joblib.load(r'H:\master\code\python\ML_HW2\model1.pkl') 
test_y2 = clf.predict(test_x)

clf = joblib.load(r'H:\master\code\python\ML_HW2\model1.pkl') 
test_y3 = clf.predict(test_x)
test_y = (test_y1+test_y2+test_y3)//2


ans = []
for i in range(len(test_y)):
    ans.append([i+1])
    ans[i].append(int(test_y[i]))

filename = str(r'G:\My Drive\code\python\ML_HW2\ans\QQ.csv')
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()
