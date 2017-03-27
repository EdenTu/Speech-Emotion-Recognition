from pymongo import MongoClient
from keras.layers import Dense, Activation
from keras.utils import np_utils
from keras.models import Sequential
model = Sequential()
model.add(Dense(input_shape=(11,),units=11))
model.add(Activation("relu"))
model.add(Dense(input_shape=(11,),units=8))
model.add(Activation("relu"))
model.add(Dense(input_shape=(8,),units=6))
model.add(Activation("relu"))
model.add(Dense(input_shape=(6,),units=4))
model.add(Activation("relu"))
model.add(Dense(input_shape=(2,),units=2))
model.add(Activation("sigmoid"))
model.add(Dense(input_shape=(1,),units=14))
# model.add(Activation("sigmoid"))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

percent_to_train=60
import numpy as np
emotion_database=MongoClient().emotion.segment_data
emotion_database_content=list(emotion_database.find({},{'_id':0}))
X=np.array([[each_emotion['0'],each_emotion['1'],each_emotion['2'],each_emotion['3'],each_emotion['4'],each_emotion['5'],each_emotion['6'],each_emotion['7'],each_emotion['8'],each_emotion['9'],each_emotion['10']] for each_emotion in emotion_database_content])
X-=np.mean(X,axis=0)
X/=np.std(X,axis=0)
train_length=int(np.ceil(percent_to_train*len(X)/100))
Y=np_utils.to_categorical(np.array([each_emotion['emotion'] for each_emotion in emotion_database_content]),14)
random=np.arange(len(X))
np.random.shuffle(random)
X=X[random]
Y=Y[random]
model.fit(X[:train_length], Y[:train_length], epochs=50, batch_size=32)
# model.fit(X, Y,validation_split=0.33,batch_size=32,epochs=120)
loss_and_metrics = model.evaluate(X[train_length:], Y[train_length:], batch_size=128)
# classes = model.predict(X[train_length:], batch_size=128)
# for each_class in range(len(classes)):
	# print(np.argmax(Y[train_length+each_class]),end=" ")
	# print(np.argmax(classes[each_class]))
print(loss_and_metrics)