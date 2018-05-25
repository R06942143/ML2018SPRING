import numpy as np
import math
import re
from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
import sys

def preprocess(text):
    return (re.sub(r'(\w{3,})s\b', r'\1',re.sub(r'(.)\1{1,}', r'\1',re.sub('[0-9]','1',text.replace(' \' ','')))))

test_path = sys.argv[1]


with open(test_path,encoding = 'utf-8') as f:
    test = f.readlines()
test_X = [preprocess(line.strip().split(",",1)[1]) for line in test]


tokenizer = Tokenizer(num_words=None,filters="\n\t")
with open('./tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
word_index = tokenizer.word_index
sequences_test = tokenizer.texts_to_sequences(test_X)
test_X_num = pad_sequences(sequences_test, maxlen=39)

model = load_model('./'+str(sys.argv[3])+'.h5py')

pred_y_prob = model.predict(test_X_num,batch_size=128)
with open(sys.argv[2], 'w') as f:
    f.write('id,label\n')
    for i in range(200000):
        f.write('%d,%d\n' %(i, np.around(pred_y_prob[i])))

