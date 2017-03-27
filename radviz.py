from pandas.tools.plotting import radviz
from pandas import DataFrame
from pandas import read_csv
from pymongo import MongoClient
from pandas import concat
from sklearn import preprocessing
import matplotlib.pyplot as PLT
min_max_scaler = preprocessing.MinMaxScaler()
emotion_database=((MongoClient().emotion).data)
data=DataFrame(list(emotion_database.find({},{'_id':0,'emotiontype':0})))
data=data.dropna(1)
print(len(data))
np_scaled = DataFrame(min_max_scaler.fit_transform(data))
np_scaled=(np_scaled.join((DataFrame(list(emotion_database.find({},{'emotiontype':1,'_id':0}))))))
radviz(np_scaled,'emotiontype')
PLT.plot()
PLT.show()