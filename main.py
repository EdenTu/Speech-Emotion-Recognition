import temp
import pitch
from scipy import stats
from pymongo import MongoClient


def get_whole_data():
	return temp.get_mfcc_feature_vector()

mfcc_data=get_whole_data()
delta_features_vector=list()

for mfcc_data_point in mfcc_data:	
	delta_features_vector.append([pitch.delta_calculator(mfcc_data_point['data'],2,index) for index in range(len(mfcc_data_point['data']))])


delta_delta_features_vector=list()
for mfcc_data_point in delta_features_vector:	
	delta_delta_features_vector.append([pitch.delta_calculator(mfcc_data_point,2,index) for index in range(len(mfcc_data_point))])
output=list()
for index in range(len(mfcc_data)):
	output+=temp.unpack(temp.segmentation(mfcc_data[index]['data']),temp.segmentation(delta_features_vector[index]),temp.segmentation(delta_delta_features_vector[index]),mfcc_data[index]['emotion'])

MongoClient().emotion.segment_Data.insert_many(output)