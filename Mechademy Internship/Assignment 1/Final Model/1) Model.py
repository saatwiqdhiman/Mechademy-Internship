
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense,RepeatVector
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,RepeatVector, LSTM, Dropout
from tensorflow.keras.layers import Bidirectional, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping


weather=pd.read_csv("Weather_data.csv")



cols = [x.replace('_','') for x in weather.columns]
weather.columns = cols
#removing "_"


cols2 = [x.replace(' ','') for x in weather.columns]
weather.columns = cols2

#removing " "

weather.columns


#converting datetime_utc's datatype to datetime datatype

weather['datetimeutc'] = pd.to_datetime(weather['datetimeutc'])

#setting datetimeutc as the index

weather = weather.set_index('datetimeutc')
weather.head(n=2)



weather = weather.dropna(thresh=(len(weather)/2) ,axis=1)
print(weather.shape)
weather.isnull().sum()



weather.fillna(method='ffill',inplace=True)
weather.isnull().sum()



from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
columns=(['conds','wdire'])
for i in columns:
    weather[i]=label.fit_transform(weather[i])
weather.head()




temp=temp.values
temp=temp.astype('float32')
temp


temp=temp.reshape(-1, 1)
scaler= MinMaxScaler(feature_range=(-1,1))
sc = scaler.fit_transform(temp)



step = 30
a= []
b=[]
for i in range(len(sc)- (step)):
    a.append(sc[i:i+step])
    b.append(sc[i+step])
a=np.asanyarray(a)
b=np.asanyarray(b)
k = 7000
atrain = a[:k,:,:]
atest = a[k:,:,:]    
btrain = b[:k]    
btest= b[k:]
print(atrain.shape)
print(atest.shape)



model = Sequential()



model.add(Conv1D(filters=128, kernel_size=2, activation='relu', input_shape=(30,1)))
model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))



model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(RepeatVector(30))



model.add(LSTM(100, activation='sigmoid'))
model.add(Dense(100, activation='sigmoid'))
model.add(Dense(1))




model.compile(loss='mse', optimizer='adam')
model.fit(atrain,btrain,epochs=100, verbose=0 )



preds_cnn1 = model.predict(atest)
preds_cnn1 = scaler.inverse_transform(preds_cnn1)


btest=np.asanyarray(btest)  
btest=btest.reshape(-1,1) 
btest = scaler.inverse_transform(btest)


btrain=np.asanyarray(btrain)  
btrain=btrain.reshape(-1,1) 
btrain = scaler.inverse_transform(btrain)

mean_squared_error(btest,preds_cnn1)

