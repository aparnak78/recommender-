#!/usr/bin/env python
# coding: utf-8

# In[1]:


from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras.layers import Bidirectional
import matplotlib.pyplot as plt
import os
import warnings

from keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from keras.models import Model

warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import numpy as np
import re

import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv('studentInfo.csv')
df.head()


# In[3]:


from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2, random_state=42)


# In[4]:


X1 = list()
X2 = list()
Y = list()
X1 = [train.course_id]
X2 = [train.id_student]
Y = [train.final_result]
print(X1)


# In[5]:


X = np.column_stack((X1, X2))
print(X)


# In[6]:


X = array(X).reshape(26074, 1, 2)


# In[7]:


model = Sequential()
model.add(LSTM(80, activation='relu', input_shape=(1, 2)))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
print(model.summary())


# In[8]:


if os.path.exists('lstm_model11.h5'):
    model = load_model('lstm_model11.h5')
else:
    model.fit(X, Y, epochs=20, validation_split=0.2, batch_size=5)


# In[9]:


X3 = [test.course_id]
X4 = [test.id_student]
Y1 = [test.final_result]
print(X3)
X5 = np.column_stack((X3, X4))


# In[10]:


X5 = array(X5).reshape(6519, 1, 2)


# In[11]:


model.evaluate(X5,Y1)


# In[12]:


#predictions = model.predict(X5,verbose=0)
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
yhat_probs = model.predict(X5, verbose=0)
# predict crisp classes for test set
yhat_classes = model.predict_classes(X5, verbose=0)
# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]
 # accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(Y1, yhat_probs)
# precision tp / (tp + fp)
precision = precision_score(Y1, yhat_probs)
# recall: tp / (tp + fn)
recall = recall_score(Y1, yhat_probs)
# f1: 2 tp / (2 tp + fp + fn)
fscore = f1_score(Y1, yhat_probs
#[print(predictions[i], test.final_result.iloc[i]) for i in range(0,10)]


# In[13]:


X6 =  np.array(list(set(df.course_id)))
X7 = np.array([2632165 for i in range(len(X6))])

X8 = np.column_stack((X6, X7))


# In[14]:


X8 = array(X8).reshape(22, 1, 2)


# In[15]:


predictions = model.predict(X8,verbose=0)
#[print(predictions[i]) for i in range(0,22)]
predictions = np.array([a[0] for a in predictions])
#[print(predictions[i]) for i in range(0,)]

recommended_course_ids = (-predictions).argsort()[:5]
print("Output recommended courses : ", X6[recommended_course_ids]) 

#recommended_course_ids


# In[16]:



print('Accuracy: %f' % accuracy)
print('Precision: %f' % precision)
print('Recall: %f' % recall)
print('F1 score: %f' % fscore)


# In[ ]:




