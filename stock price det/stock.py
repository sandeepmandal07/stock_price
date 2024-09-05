import math#use for mathematical operations
import numpy as np#use for operation in array
import pandas as pd#use to manipulated numerical data and time series
from sklearn.preprocessing import MinMaxScaler#use to scale values to a specified range
from keras.models import Sequential#sequentially arrange stack of layers
from keras.layers import Dense, LSTM#use for neural networks
import matplotlib.pyplot as plt#used for plotting graphs
df=pd.read_csv('C:/Users/Akash/Downloads/AAPL.csv')
plt.figure(figsize=(10,6))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('no of stocks',fontsize=38)
plt.ylabel('Close Price USD ($)',fontsize=38)
plt.show()
#creating a new dataframe with only the close column
data=df.filter(['Close'])
#converting the dataframe to a numpy array
dataset=data.values
#getting the number of rows to train the model on
training_data_len=math.ceil(len(dataset)*.8)
training_data_len
#scaling the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data=scaler.fit_transform(dataset)
scaled_data
#create the training data set
#create teh scaled training data set
train_data=scaled_data[0:training_data_len,:]
#split the data into x_train and y_train data sets
x_train =[]
y_train=[]

for i in range(60,len(train_data)):
  x_train.append(train_data[i-60:i,0])
  y_train.append(train_data[i,0])
  if i<=60:
    print(x_train)
    print(y_train)
    print()


#converting the x_train and y_train to numpy arrays
x_train,y_train=np.array(x_train),np.array(y_train)
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
x_train.shape
#building the LSTM model 1
model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(50,return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
#building the LSTM model
model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(100,return_sequences=False))
model.add(Dense(12))
model.add(Dense(1))

#building the LSTM mode
model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(80,return_sequences=False))
model.add(Dense(12))
model.add(Dense(4))
model.add(Dense(1))
#building the LSTM model 4
model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(50,return_sequences=False))
model.add(Dense(12))
model.add(Dense(2))
model.add(Dense(1)) 
model.compile(optimizer='adam',loss='mean_squared_error')
#training the model
model.fit(x_train,y_train,batch_size=1,epochs=1)
#creating the testing dataset
#creating a new array containing scaled values from index 2155 to 2768
test_data=scaled_data[training_data_len-60:,:]
#creating the dataset x_test and y_test
x_test=[]
y_test=dataset[training_data_len:,:]
for i in range(60,len(test_data)):
  x_test.append(test_data[i-60:i,0])
x_test=np.array(x_test)
#reshaping the data
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
#getting the model predicted price values
predictions=model.predict(x_test)
predictions=scaler.inverse_transform(predictions)
#getting the root mean squared error(RMSE)
rmse=np.sqrt(np.mean(predictions-y_test)**2)
rmse
#plotting the data
train =data[:training_data_len]
valid=data[training_data_len:]
valid['Predictions']=predictions
#visualizing the data
plt.figure(figsize=(30,15))
plt.title('Model')
plt.xlabel('Date',fontsize=38)
plt.ylabel('Close Price USD($)',fontsize=38)
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Train','Val','Predictions'],loc='lower right')
plt.show()