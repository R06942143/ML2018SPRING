import numpy as np
import sys
import csv

w = [0.05652708,
0.03698634,
-0.14013027,
0.19389618,
-0.09844684,
0.07997062,
-0.1131139,
0.18495971,
0.77063509
]
b= [0.33603384]

dataQ = np.genfromtxt(sys.argv[1],delimiter=',',dtype=float)
dataQ = np.delete(dataQ,range(2),axis=1)
dataQ = np.nan_to_num(dataQ)
dell_ =[]
for i in range(len(dataQ)):
    if(i%18 !=9):
        dell_.append(i)

dataQ = np.delete(dataQ,dell_,0)
for i in range(len(dataQ)):
    for j in range(len(dataQ[0])):
        if(dataQ[i][j]>110 or dataQ[i][j]<5):
            if(j != len(dataQ[0])-1):
                dataQ[i][j] = dataQ[i][j+1]
            else:
                dataQ[i][j] = dataQ[i][j-1]


##############submission.csv###################################
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

filename = str(sys.argv[2])
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i]) 
text.close()