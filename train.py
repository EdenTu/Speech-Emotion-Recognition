from pymongo import MongoClient
from keras.layers import Dense, Activation

from keras.models import Sequential
model = Sequential()
model.add(Dense(input_shape=(70916,),units=30000))
model.add(Activation("relu"))
model.add(Dense(input_shape=(30000,),units=2000))
model.add(Activation("relu"))
model.add(Dense(units=1))
model.add(Activation("softmax"))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

percent_to_train=10
import numpy as np
emotion_database=MongoClient().emotion.segment_data
emotion_database_content=list(emotion_database.find({},{'_id':0}))
X=np.array([[each_emotion['0'],each_emotion['1'],each_emotion['2'],each_emotion['3'],each_emotion['4'],each_emotion['5'],each_emotion['6'],each_emotion['7'],each_emotion['8'],each_emotion['9'],each_emotion['10']] for each_emotion in emotion_database_content]).T
train_length=int(np.ceil(percent_to_train*len(X)/100))
Y=np.array([each_emotion['emotion'] for each_emotion in emotion_database_content])
random=np.arange(len(X))
np.random.shuffle(random)
X=X[random]
Y=Y[random]
model.fit(X, Y,validation_split=0.33,epochs=150,batch_size=32)
