import tensorflow as tf

from tensorflow import keras
from keras import Sequential
from keras.layers import Dense
from sklearn import metrics
import seaborn as sns
import pandas as pd
import numpy as np
df=pd.read_csv('C:/Users/sayak/OneDrive/Documents/Projects/IITM_IPL_Hackathon/final_ipl_iit.csv')
#print('printing engineering')
X=df.iloc[:,0:52].values
y=df.iloc[:,-1].values

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=50)

model = Sequential()

model.add(Dense(106, activation='relu'))
model.add(Dense(212, activation='relu'))
model.add(Dense(106, activation='relu'))
model.add(Dense(53, activation='relu'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(
    X_train,
    y_train,
    epochs=100,
    shuffle=True,
    verbose=2
)

prediction = model.predict(X_test)

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))
