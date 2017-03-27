from pymongo import MongoClient
from keras.layers import Dense, Activation
from keras.utils import np_utils
from keras.models import Sequential
model = Sequential()
model.add(Dense(input_shape=(33,),units=750))
model.add(Activation("softmax"))
model.add(Dense(units=250))
model.add(Activation("relu"))
model.add(Dense(units=250))
model.add(Activation("relu"))
model.add(Dense(units=256))
model.add(Activation("relu"))
model.add(Dense(units=16))
# model.add(Activation("sigmoid"))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

percent_to_train=60
import numpy as np
emotion_database=MongoClient().emotion.segment_Data
emotion_database_content=list(emotion_database.find({},{'_id':0}))
X=np.array([each_emotion['data'] for each_emotion in emotion_database_content])
X-=np.mean(X,axis=0)
X/=np.std(X,axis=0)
train_length=int(np.ceil(percent_to_train*len(X)/100))
Y=np_utils.to_categorical(np.array([each_emotion['emotion'] for each_emotion in emotion_database_content]),16)
random=np.arange(len(X))
np.random.shuffle(random)
length=0

		# length+=1
# print(length)
# X=X[random]
# Y=Y[random]
# model.fit(X[:train_length], Y[:train_length], epochs=50, batch_size=32)

result=list()
for index in range(16):
	result.append(list())


model.fit(X[:-100], Y[:-100],batch_size=32,epochs=50)

# loss_and_metrics = model.evaluate(X[train_length:], Y[train_length:], batch_size=128)
classes = model.predict(X[(len(X)-100):], batch_size=128)


for each_class in classes:
	for index in range(16):
		result[index].append(each_class[index])

import matplotlib.pyplot as pyplot
labels=['happy','neutral','disgust','anger',    'anxiety',    'despair',    'sadness',    'elation',    'interest',    'boredom',    'pride',    'contempt',    'shame',    'panic',    'hot',    'cold']
x=[i for i in range(100)]
for data in range(16):
	pyplot.plot(x,result[data],label=labels[data])
pyplot.legend()
pyplot.show()
# print(loss_and_metrics)