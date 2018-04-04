from sklearn.svm import SVC
import numpy as np
import csv
from sklearn.externals import joblib
import sys


de_feature =[10]

test_x  = np.genfromtxt(sys.argv[5],delimiter=',')
test_x  = np.delete(test_x,0,0)

test_x = np.delete(test_x,de_feature,1)

clf = joblib.load('./clf.pkl')

test_y = clf.predict(test_x)


ans = []
for i in range(len(test_y)):
    ans.append([i+1])
    ans[i].append(int(test_y[i]))

filename = str(sys.argv[6])
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(ans)):
    s.writerow(ans[i]) 

text.close()