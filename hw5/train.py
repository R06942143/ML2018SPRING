import numpy as np
import math
import re
from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import pickle 
import gensim
from gensim.models import word2vec
import sys


# train_label_path = './training_label.txt'
# test_path = './testing_data.txt'
train_label_path = sys.argv[1]
# test_path = './testing_data.txt'
train_nolabel_path = sys.argv[2]

with open(train_nolabel_path,encoding = 'utf-8') as f:
    train_no = f.readlines()
trian_no_X = [str(re.sub(r'(\w{3,})s\b', r'\1',re.sub(r'(.)\1{1,}', r'\1',re.sub('[0-9]','1',seg.replace(' \' ','').replace('\n',''))))) for seg in train_no]

with open(train_label_path,encoding = 'utf-8') as f:
    train = f.readlines()
# train_X = [seg.strip().split(" +++$+++ ")[1].replace(' \' ','') for seg in train]
# train_X = [re.sub(r'(.)\1{1,}', r'\1',re.sub('[0-9]','1',seg.strip().split(" +++$+++ ")[1].replace(' \' ',''))) for seg in train]
train_X = [str(re.sub(r'(\w{3,})s\b', r'\1',re.sub(r'(.)\1{1,}', r'\1',re.sub('[0-9]','1',seg.strip().split(" +++$+++ ")[1].replace(' \' ',''))))) for seg in train]
train_y = [seg.strip().split(" +++$+++ ")[0].replace(' \' ','') for seg in train]

# with open(test_path,encoding = 'utf-8') as f:
#     test = f.readlines()
# # test_X = [seg.strip().split(",",1)[1].replace(' \' ','') for seg in test]
# # test_X = [re.sub(r'(.)\1{1,}', r'\1',re.sub('[0-9]','1',seg.strip().split(",",1)[1].replace(' \' ',''))) for seg in test]
# test_X = [str(re.sub(r'(\w{3,})s\b', r'\1',re.sub(r'(.)\1{1,}', r'\1',re.sub('[0-9]','1',seg.strip().split(",",1)[1].replace(' \' ',''))))) for seg in test]



tokenizer = Tokenizer(num_words=None,filters="\n\t")
with open('./tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# tokenizer.fit_on_texts(train_X)
sequences = tokenizer.texts_to_sequences(train_X)
# sequences_test = tokenizer.texts_to_sequences(test_X)
sequences_trainno = tokenizer.texts_to_sequences(trian_no_X)
# test_X_num = pad_sequences(sequences_test, maxlen=39)
word_index = tokenizer.word_index


total_corpus = train_X
total_corpus = [ sent.split(" ") for sent in total_corpus]
emb_size = 128
w2v_model = gensim.models.Word2Vec(total_corpus, size=emb_size, window=5, min_count=1, workers=8)
# w2v_mode.load('./word2vec')


embedding_matrix = np.zeros((len(word_index), emb_size))
for word, i in word_index.items():
    try:
        embedding_vector = w2v_model.wv[word]
        embedding_matrix[i] = embedding_vector
    except:
        print(word)

max_length = np.max([len(i) for i in sequences])

train_y_numeric = np.array(train_y,dtype=int)

train_X_num = pad_sequences(sequences, maxlen=39)
train_no_X  = pad_sequences(sequences_trainno,maxlen=39)
model = Sequential()


model.add(Embedding(len(word_index),output_dim= emb_size,
                            weights=[embedding_matrix],
                            input_length=max_length,
                            trainable=False))
model.add(Bidirectional(LSTM(128,activation='tanh',return_sequences=True,dropout=0.3)))
model.add(BatchNormalization())
model.add(Bidirectional(LSTM(64,activation='tanh',dropout=0.3)))
model.add(BatchNormalization())
model.add(Dense(32,activation ='relu'))
model.add(Dropout(0.3))
model.add(Dense(16,activation ='relu'))
model.add(Dropout(0.3))
model.add(Dense(1,activation ='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',optimizer ='Adam',metrics=['acc'])




X=train_X_num[:180000]
Y=train_y_numeric[:180000]
v_X = train_X_num[180000:]
v_Y = train_y_numeric[180000:]
callbacks = ModelCheckpoint('./model-semi.h5py',monitor='val_acc',save_best_only=True)
# model = load_model('./model.h5py') in the first. you have to load the supervised model
no_Y = np.around(model.predict(train_no_X,batch_size = 512))
total_Y = np.concatenate((Y,np.reshape(no_Y,len(no_Y))))
total_X = np.concatenate((X,train_no_X),axis = 0)
for i in range(50):
    model.fit(total_X,total_Y,validation_data= (v_X,v_Y),epochs= 1,
                    batch_size= 512,callbacks=[callbacks],
                    verbose=1,shuffle = True)
    no_Y = np.around(model.predict(train_no_X,batch_size = 512))
    total_Y = np.concatenate((Y,np.reshape(no_Y,len(no_Y))))

# pred_y_prob = model.predict(test_X_num,batch_size=512)
# with open('./submission.csv', 'w') as f:
#     f.write('id,label\n')
#     for i in range(200000):
#         f.write('%d,%d\n' %(i, np.around(pred_y_prob[i])))




















