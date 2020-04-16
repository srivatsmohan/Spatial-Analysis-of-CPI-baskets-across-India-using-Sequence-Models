#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv(r'C:\Users\SRIVATS MOHAN\Documents\Acads\Internship\Data\Rice/'+'Rice_All.csv')
df.head(10)


# In[3]:


df['Arrival_Date'] = pd.to_datetime(df["Arrival_Date"])
df_idx = df.set_index(["Arrival_Date"], drop=True)
df_idx.head(5)


# In[4]:


df.shape


# In[5]:


df_AndPra=df[df.State=='Andhra Pradesh']
df_AndPra.drop(['State'],axis=1,inplace=True)
df_AndPra = df_AndPra[df_AndPra['Arrival_Date']<='2018-10-31']
#f_AndPra = df_AndPra[df_AndPra['Arrival_Date']>='2013-01-01']
df_AndPra_idx = df_AndPra.set_index(["Arrival_Date"], drop=True)
df_AndPra_idx.head(5)


# In[6]:


df_AndPra.describe()


# In[7]:


df_AndPra.nunique()


# In[8]:


df_AndPra_idx = df_AndPra_idx.sort_index(axis=1, ascending=True)
df_AndPra_idx = df_AndPra_idx.iloc[::-1]


# In[9]:


data = df_AndPra_idx[['Modal Price']]
data.plot(y='Modal Price',figsize=(24,18))
data.sort_index(axis=0, ascending=True,inplace=True)
plt.show()


# In[10]:


data.index.values


# In[11]:


data


# In[12]:


diff = data.index.values[-1] - data.index.values[0]
days = diff.astype('timedelta64[D]')

days = days / np.timedelta64(1, 'D')
years = int(days/365)
print("Total data: %d years"%years)
print("80 percent data = 2010 to %d"%(2013 + int(0.8*years)))


# In[13]:


split_date = pd.Timestamp('01-01-2017')

train = data.loc[:split_date]
test = data.loc[split_date:]

ax = train.plot(figsize=(10,12))
test.plot(ax=ax)
plt.legend(['train', 'test'])
plt.show()


# In[14]:


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
train_sc = sc.fit_transform(train)
test_sc = sc.transform(test)


# In[15]:


X_train = train_sc[:-1]
y_train = train_sc[1:]

X_test = test_sc[:-1]
y_test = test_sc[1:]


# In[16]:


from sklearn.metrics import r2_score

def adj_r2_score(r2, n, k):
    return 1-((1-r2)*((n-1)/(n-k-1)))


from sklearn.svm import SVR
regressor = SVR(kernel='rbf')

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
y_pred = y_pred.reshape(-1,1)
y_test_tfm = sc.inverse_transform(y_test)
y_pred_tfm = sc.inverse_transform(y_pred)
plt.plot(y_test_tfm)
plt.plot(y_pred_tfm)

print('R-Squared: %f'%(r2_score(y_test_tfm, y_pred_tfm)))


# In[18]:


from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import load_model


# In[19]:


K.clear_session()

model = Sequential()
model.add(Dense(12, input_dim=1, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
history = model.fit(X_train, y_train, epochs=200, batch_size=1, verbose=1, callbacks=[early_stop], shuffle=False)


# In[20]:


y_pred_test_ann = model.predict(X_test)
y_train_pred_ann = model.predict(X_train)
print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred_ann)))
r2_train = r2_score(y_train, y_train_pred_ann)
print("The Adjusted R2 score on the Train set is:\t{:0.3f}\n".format(adj_r2_score(r2_train, X_train.shape[0], X_train.shape[1])))
print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred_test_ann)))
r2_test = r2_score(y_test, y_pred_test_ann)
print("The Adjusted R2 score on the Test set is:\t{:0.3f}".format(adj_r2_score(r2_test, X_test.shape[0], X_test.shape[1])))


# In[21]:


model.save('ANN_NonShift.h5')


# In[22]:


X_tr_t = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_tst_t = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])


# In[23]:


from keras.layers import LSTM
K.clear_session()
model_lstm = Sequential()
model_lstm.add(LSTM(7, input_shape=(1, X_train.shape[1]), activation='relu', kernel_initializer='lecun_uniform', return_sequences=False))
model_lstm.add(Dense(1))
model_lstm.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
history_model_lstm = model_lstm.fit(X_tr_t, y_train, epochs=200, batch_size=10, verbose=1, shuffle=False, callbacks=[early_stop])


# In[24]:


y_pred_test_lstm = model_lstm.predict(X_tst_t)
y_train_pred_lstm = model_lstm.predict(X_tr_t)
print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred_lstm)))
r2_train = r2_score(y_train, y_train_pred_lstm)
print("The Adjusted R2 score on the Train set is:\t{:0.3f}\n".format(adj_r2_score(r2_train, X_train.shape[0], X_train.shape[1])))
print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred_test_lstm)))
r2_test = r2_score(y_test, y_pred_test_lstm)
print("The Adjusted R2 score on the Test set is:\t{:0.3f}".format(adj_r2_score(r2_test, X_test.shape[0], X_test.shape[1])))


# In[25]:


model_lstm.save('LSTM_NonShift.h5')


# In[26]:


model_ann = load_model('ANN_NonShift.h5')
model_lstm = load_model('LSTM_NonShift.h5')


# In[27]:


score_ann= model_ann.evaluate(X_test, y_test, batch_size=1)
score_lstm= model_lstm.evaluate(X_tst_t, y_test, batch_size=1)


# In[28]:


print('ANN: %f'%score_ann)
print('LSTM: %f'%score_lstm)


# In[29]:


y_pred_test_ANN = model_ann.predict(X_test)
y_pred_test_LSTM = model_lstm.predict(X_tst_t)
y_train_pred_lstm = model_lstm.predict(X_tr_t)


# In[31]:


plt.plot(y_test, label='True')
plt.plot(y_pred_test_ANN, label='ANN')
plt.title("ANN's_Prediction")
plt.xlabel('Observation')
plt.ylabel('Price_Scaled')
plt.legend()
plt.show()


# In[32]:


plt.plot(y_test, label='True')
plt.plot(y_pred_test_LSTM, label='LSTM')
plt.title("LSTM's_Prediction Forecast")
plt.xlabel('Observation')
plt.ylabel("Price_Scaled")
plt.legend()
plt.show()


# In[33]:


plt.plot(y_train, label='True')
plt.plot(y_train_pred_lstm, label='LSTM')
plt.title("LSTM's_Prediction on Training")
plt.xlabel('Observation')
plt.ylabel('Price_Scaled')
plt.legend()
plt.show()


# In[ ]:




