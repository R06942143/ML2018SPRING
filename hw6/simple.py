import numpy as np
import keras.backend as K
from keras.regularizers import l2
from keras.models import *
from keras.layers import *
from keras.callbacks import *
import numpy as np
import random
import pickle
import sys


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.pow(y_true - y_pred, 2)))

def get_model(user,item,latent_dim=30):
    user_input  = Input(shape=(1,))
    item_input  = Input(shape=(1,))
    user_vec    = Embedding(10000,latent_dim,embeddings_initializer = 'random_normal',embeddings_regularizer = l2(0.000005))(user_input)
    user_vec    = Flatten()(user_vec)
    item_vec    = Embedding(10000,latent_dim,embeddings_initializer = 'random_normal',embeddings_regularizer = l2(0.000005))(item_input)
    item_vec    = Flatten()(item_vec)
    user_bias   = Embedding(10000,1,embeddings_initializer = 'zero')(user_input)
    user_bias   = Flatten()(user_bias)
    item_bias   = Embedding(10000,1,embeddings_initializer = 'zero')(item_input)
    item_bias   = Flatten()(item_bias)
    output       = Dot(axes = 1,normalize= True)([user_vec,item_vec])
    output       = Add()([output,user_bias,item_bias])
    output = Lambda(lambda x: x + K.constant(3.5817, dtype=K.floatx()))(output)
    model       = Model([user_input,item_input],output)
    model.compile(loss='mse',optimizer='adam',metrics=[rmse])
    return model

def get_model_dnn(user,item,latent_dim=16):
    user_input  = Input(shape=(1,))
    item_input  = Input(shape=(1,))
    user_vec    = Embedding(len(user),latent_dim,embeddings_initializer = 'random_normal',embeddings_regularizer = l2(0.000005))(user_input)
    user_vec    = Flatten()(user_vec)
    item_vec    = Embedding(len(item),latent_dim,embeddings_initializer = 'random_normal',embeddings_regularizer = l2(0.000005))(item_input)
    item_vec    = Flatten()(item_vec)
    merge_vec   = Concatenate()([user_vec,item_vec])
    hidden      = Dense(150,activation = 'relu')(merge_vec)
    hidden      = Dense(50,activation = 'relu')(hidden)
    output      = Dense(1)(hidden)
    model       = Model([user_input,item_input],output)
    model.compile(loss='rmse',optimizer='adam',metrics=[rmse])
    return model

b = np.genfromtxt(sys.argv[1],delimiter = ',',skip_header=1)

test_X = b[:,1:]
model = load_model('./simple.h5', custom_objects={'rmse': rmse})
outcome  =  model.predict([test_X[:,0],test_X[:,1]])

with open(str(sys.argv[2]), 'w') as f:
    f.write('TestDataID,Rating\n')
    for i in range(1,len(outcome)+1):
        f.write('%d,%f\n' %(i, outcome[i-1]))

