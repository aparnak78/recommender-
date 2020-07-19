from numpy import array

from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM

from keras.models import Model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate


import pandas as pd
import numpy as np
import re
import pickle

import matplotlib.pyplot as plt
df=pd.read_csv('studentInfo.csv')
import matplotlib.pyplot as plt
import os
import warnings

from keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from keras.models import Model


from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2)
X1 = list()
X2 = list()
Y = list()
X1 = [train.course_id]
X2 = [train.id_student]
Y = [train.final_result]
X = np.column_stack((X1, X2))
X = array(X).reshape(26074, 1, 2)
from keras.layers import GRU
model = Sequential()
model.add(GRU(80, activation='relu', input_shape=(1, 2)))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

if os.path.exists('lstm_model7.h5'):
    model = load_model('lstm_model7.h5')
else:
    model.fit(X, Y, epochs=20, validation_split=0.2, batch_size=5)


pickle.dump(model, open('model.pkl','wb'))