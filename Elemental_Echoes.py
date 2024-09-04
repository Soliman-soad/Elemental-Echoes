import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('weather_data.csv')
X = dataset.iloc[:,[1,2,3,4,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]].values
Y = dataset.iloc[:,-1].values
print(X)

Y = Y.reshape(-1,1)
#determine missing value
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
X = imputer.fit_transform(X)
Y = imputer.fit_transform(Y)

print(Y)
#label all the data
from sklearn.preprocessing import LabelEncoder
le1 = LabelEncoder()
X[:,0] = le1.fit_transform(X[:,0])
le2 = LabelEncoder()
X[:,4] = le2.fit_transform(X[:,4])
le3 = LabelEncoder()
X[:,6] = le3.fit_transform(X[:,6])
le4 = LabelEncoder()
X[:,7] = le4.fit_transform(X[:,7])
le5 = LabelEncoder()
X[:,-1] = le5.fit_transform(X[:,-1])
le6 = LabelEncoder()
Y[:,-1] = le6.fit_transform(Y[:,-1])
#label independent data
print(X)
#label dependent data
print(Y)

Y = np.array(Y,dtype=float)
print(Y)

#scale all the data for making a good prediction
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
print(X)
     
#making the train and test set value 
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

#now making a prediction in probability with ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

#initialize the ANN model
classifire=Sequential()
#making the first hidden layer:
classifire.add(Dense(10,kernel_initializer='uniform',activation='relu',input_dim=19))
#making the 2nd hidden layer:
classifire.add(Dense(10,kernel_initializer='uniform',activation='relu'))
#intialize the output layer:
classifire.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))  # Output layer should have 1 neuron

#compile the artificial neural network:
classifire.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#fit the ANN with the training set 
classifire.fit(X_train,Y_train,batch_size=10,epochs=100)

#making the prediction and evaluate the model


#predicting the test set result
y_pred=classifire.predict(X_test)
y_pred=(y_pred>.5)

# now i gonna use the confusion metrics for checking the model validation 

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(X_test, y_pred)