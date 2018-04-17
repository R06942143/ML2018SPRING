import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dropout,Activation,Flatten,Conv2D,MaxPooling2D,ZeroPadding2D
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop,Adagrad,Adam
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator



print('read')
with open(str(sys.argv[1]),'rb') as f:
    clean_lines = (line.replace(b',',b' ') for line in f)
    train_xn = np.genfromtxt(clean_lines,dtype= float,delimiter = ' ',skip_header= 1)

train_y = train_xn[:,0]

# m = np.mean(train_xn[:,1:],axis= 1)
# s = np.std(train_xn[:,1:],axis= 1)
# for i in range(1,len(train_xn[0])):
#     train_xn[:,i] = (train_xn[:,i] - m)/s
train_xn[:,1] /= 255.0
train_x = train_xn[:,1:].reshape(len(train_xn),48,48,1)

# train_x /=train_x


ans = np.zeros((len(train_y),7))

for i in range(len(train_y)):
    ans[i][int(train_y[i])] = 1


t_x  = train_x[:23000]
v_x  = train_x[23000:]

t_y  = ans[:23000]
v_y  = ans[23000:]

#######################split##################################
print('complete')


# datagen = ImageDataGenerator( rotation_range=1,\
#     width_shift_range=0.2, height_shift_range=0.2, \
#     rescale=1./255, shear_range=0.2, zoom_range=0.2,\
#     horizontal_flip=True, fill_mode='nearest')
datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=[0.8, 1.2],
            shear_range=0.2,
            horizontal_flip=True)

f1 = 64
f2 = 128
f3 = 256
f4 = 512
f5 = 512
Dens=1000

model=Sequential()
model.add(Conv2D(filters=f1,kernel_size=(3,3),input_shape=(48,48,1),padding='same')) #225*44*44
model.add(BatchNormalization())
model.add(Activation('selu'))
model.add(Conv2D(filters=f1,kernel_size=(3,3),padding='same')) #225*44*44
model.add(BatchNormalization())
model.add(Activation('selu'))
model.add(MaxPooling2D(pool_size=(2,2))) #225*22*22
model.add(Dropout(0.2))


model.add(Conv2D(filters=f2,kernel_size=(3,3),padding='same')) #225*44*44
model.add(BatchNormalization())
model.add(Activation('selu'))
model.add(Conv2D(filters=f2,kernel_size=(3,3),padding='same')) #225*44*44
model.add(BatchNormalization())
model.add(Activation('selu'))
model.add(MaxPooling2D(pool_size=(2,2))) #225*22*22
model.add(Dropout(0.2))

model.add(Conv2D(filters=f3,kernel_size=(3,3),padding='same')) #225*44*44
model.add(BatchNormalization())
model.add(Activation('selu'))
model.add(Conv2D(filters=f3,kernel_size=(3,3),padding='same')) #225*44*44
model.add(BatchNormalization())
model.add(Activation('selu'))
model.add(Conv2D(filters=f3,kernel_size=(3,3),padding='same')) #225*44*44
model.add(BatchNormalization())
model.add(Activation('selu'))
model.add(Conv2D(filters=f3,kernel_size=(3,3),padding='same')) #225*44*44
model.add(BatchNormalization())
model.add(Activation('selu'))
model.add(MaxPooling2D(pool_size=(2,2))) #225*22*22
model.add(Dropout(0.2))


model.add(Conv2D(filters=f4,kernel_size=(3,3),padding='same')) #225*44*44
model.add(BatchNormalization())
model.add(Activation('selu'))
model.add(Conv2D(filters=f4,kernel_size=(3,3),padding='same')) #225*44*44
model.add(BatchNormalization())
model.add(Activation('selu'))
model.add(Conv2D(filters=f4,kernel_size=(3,3),padding='same')) #225*44*44
model.add(BatchNormalization())
model.add(Activation('selu'))
model.add(Conv2D(filters=f4,kernel_size=(3,3),padding='same')) #225*44*44
model.add(BatchNormalization())
model.add(Activation('selu'))
model.add(MaxPooling2D(pool_size=(2,2))) #225*22*22
model.add(Dropout(0.2))



model.add(Conv2D(filters=f5,kernel_size=(3,3),padding='same')) #225*44*44
model.add(BatchNormalization())
model.add(Activation('selu'))
model.add(Conv2D(filters=f5,kernel_size=(3,3),padding='same')) #225*44*44
model.add(BatchNormalization())
model.add(Activation('selu'))
model.add(Conv2D(filters=f5,kernel_size=(3,3),padding='same')) #225*44*44
model.add(BatchNormalization())
model.add(Activation('selu'))
model.add(Conv2D(filters=f5,kernel_size=(3,3),padding='same')) #225*44*44
model.add(BatchNormalization())
model.add(Activation('selu'))
model.add(MaxPooling2D(pool_size=(2,2))) #225*22*22
model.add(Dropout(0.2))


model.add(Flatten())
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Activation('selu'))
model.add(Dense(4096))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Activation('selu'))
model.add(Dense(4096))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Activation('selu'))
model.add(Dense(1000))
model.add(BatchNormalization())
model.add(Activation('selu'))
model.add(Dense(7,activation='softmax'))

print('go')
model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])
checkpoint = ModelCheckpoint("\-{epoch:02d}-{val_acc:.2f}.h5py", monitor='val_acc', verbose=1, save_best_only=True, mode='max',period=1)

batch_size=128


model.fit_generator(
            datagen.flow(t_x, t_y, batch_size=batch_size), 
            steps_per_epoch=3*len(train_x)//batch_size,
            validation_data=(v_x,v_y),
            epochs=300, callbacks=[checkpoint])

