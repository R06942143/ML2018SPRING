import io
import os
import sys
import re 
import numpy as np
import csv
from keras.models import Sequential,load_model
from keras.layers import Dropout,Activation,Flatten,Conv2D,MaxPooling2D,ZeroPadding2D
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop,Adagrad,Adam
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator



print('read')
with open(sys.argv[1],'rb') as f:
    clean_lines = (line.replace(b',',b' ') for line in f)
    train_xn = np.genfromtxt(clean_lines,dtype= float,delimiter = ' ',skip_header= 1)


train_x = train_xn[:,1:].reshape(len(train_xn),48,48,1)


if(sys.argv[3] == 'public'):
    fff = './public.h5py'
elif(sys.argv[3] == 'private'):
    fff = './private.h5py'
else:
    print('I cannot tell which model u want, so i will load public.h5py')
    fff = './public.h5py'
model = Sequential()
model = load_model(fff)
# model.summary()

print("predicting")
result = np.argmax(model.predict(train_x), axis=1)
print("writting")
pre = []

for i in range(result.shape[0]):
    pre.append([str(i)])
    pre[i].append(result[i])

filename = sys.argv[2]
text = open(filename, "w+")
s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(["id", "label"])
for i in range(len(pre)):
    s.writerow(pre[i])

text.close()
