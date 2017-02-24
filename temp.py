from scipy.io import wavfile
from scipy import signal
from scipy.fftpack import dct
from enum import Enum
import numpy as np
import matplotlib.pyplot as PLT
import pymongo
from pymongo import MongoClient
from bson.binary import Binary
emotion_database=(MongoClient()).emotion
audio_file_name_to_read="read_file.wav"
transcript_file_name_to_read="read_file.txt"
parsed_audio_Data=list()
class emotion_type(Enum):
    happy=0
    neutral=1
    disgust=2
    anger=3
    anxiety=4
    despair=5
    sadness=6
    elation=7
    interest=8
    boredom=9
    pride=10
    contempt=11
    shame=12
    panic=13
class audio_cluster:
    def __init__(self,start_index,final_index,data,emotion_type):
        self.starting_index=start_index
        self.final_index=final_index
        self.content=data
        self.emotion_category=emotion_type
        self.audio_bytes=None
    @classmethod
    def fromstring(cls, name):
        return cls.__dict__[name]
def parse_enum_type(data_to_parse):
    return emotion_type.__dict__.get(data_to_parse)
def parse_transcript():
    #format should be start_time end_time data
    with open(transcript_file_name_to_read) as f:
        file=f.readlines()
    index=10
    global parsed_audio_Data
    while index<len(file):
        data=file[index].split(" ",3)
        content=data[3].split(",",1)
        emotion_cat=content[0].split(" ")[0]
        parsed_emotion=parse_enum_type(emotion_cat)
        parsed_audio_Data.append(audio_cluster(int(frequency*float(data[0])),int(frequency*float(data[1])),content[0].rstrip("\n"),parsed_emotion))
        if parsed_emotion is not None or len(data)==4:
       		 index=index+2
def voice_between_two_points(index):
    global audio_data,parsed_audio_Data
    parsed_audio_Data[index].audio_bytes=audio_data[parsed_audio_Data[index].starting_index:parsed_audio_Data[index].final_index]
def audio_reading(file_name):
    global frequency
    global audio_data
    frequency,audio_data=wavfile.read("read_file.wav")
def write_to_file(index):
    global parsed_audio_Data
    if parsed_audio_Data[index].audio_bytes is not None:
        wavfile.write(parsed_audio_Data[index].content+".wav",frequency,parsed_audio_Data[index].audio_bytes)
        #print("File saved with name "+parsed_audio_Data[index].content+".wav")
    else:
        print("Data Not Found for "+parsed_audio_Data[index].content)

def pre_emphasis(data,pf):
    return np.append(data[0],data[1:]-pf*data[:-1])    

def framing(data,frame_size,frame_stride,frequency_of_signal):  #frame_size size of each frame
    length_of_each_frame=int(round(frame_size*frequency_of_signal))
    unique_content_in_each_frame=int(round(frame_stride*frequency_of_signal))
    """  for first and last frame unique content = (length_of_each_frame/2)+unique_content_in_each_frame
         for others its just unique_content_in_each_frame
         Therefore no_of_frames= (total_signal_length-length_of_each_frame)/unique_content_in_each_frame"""
    no_of_frames=int(np.ceil(np.abs(data.size-length_of_each_frame)/unique_content_in_each_frame))+1
    new_length=((no_of_frames-1)*unique_content_in_each_frame)+length_of_each_frame
    padding=np.zeros(new_length-data.size)
    #print(padding.size)
    data=np.append(data,padding)
    #print(data.size)

    return data[np.tile(np.arange(0,length_of_each_frame),(no_of_frames,1))+np.tile(range(0,(unique_content_in_each_frame)*no_of_frames,unique_content_in_each_frame),(length_of_each_frame,1)).T]

def windowing(data):
    return data*np.hamming(data[0].size)

def fft_power_spectrum(data,No_of_points):
    #print("some   "+str(data[0].size))
    #print(np.fft.rfft(data,No_of_points)[0].size)
    return (np.absolute(np.fft.rfft(data,No_of_points))**2)/No_of_points

def triangular_filter_banks(frequency,no_of_filters,No_of_points,data):
    low_freq=0
    high_freq_in_mel=2595*np.log10(1+frequency/1400)
    range_of_freq_in_mel=np.linspace(low_freq,high_freq_in_mel,no_of_filters+2)
    range_of_freq_in_hz=((10**(range_of_freq_in_mel/2595))-1)*700
    bins=np.floor((No_of_points+1)*range_of_freq_in_hz/frequency)
    np.zeros(no_of_filters)
    frequency_bank=np.zeros((no_of_filters,int(np.floor((No_of_points/2)+1))))
    for m in range(1,no_of_filters):
        left_side_of_central_freq=int(bins[m-1])
        central_freq=int(bins[m])
        right_side_of_central_freq=int(bins[m+1])
        for each_frequency in  range(left_side_of_central_freq,central_freq):
            frequency_bank[m,each_frequency]=(each_frequency-bins[m-1])/(bins[m]-bins[m-1])
        for each_frequency in range(central_freq,right_side_of_central_freq):
            frequency_bank[m,each_frequency]=(bins[m+1]-each_frequency)/(bins[m+1]-bins[m])
    triangular_filter_bank=np.dot(data,frequency_bank.T)
    triangular_filter_bank=np.where(triangular_filter_bank==0,np.finfo(float).eps,triangular_filter_bank)
    return 20*np.log10(triangular_filter_bank)

def mfcc(data):
    mfcc = dct(data, type=2, axis=1, norm='ortho')[:, 1 : (12)] 
    return mfcc

audio_reading(audio_file_name_to_read)
parse_transcript()
# print(len(parsed_audio_Data))

index=0
while index<len(parsed_audio_Data):
	data={};
	data["emotion_type"]=str(parsed_audio_Data[index].emotion_category)
	voice_between_two_points(index)
	normalization_parameter = (np.ceil(np.log2(np.amax(parsed_audio_Data[index].audio_bytes[:,1]))))
	pre_emphasized_data=pre_emphasis(parsed_audio_Data[index].audio_bytes[:,1]/(2**normalization_parameter),0.97)       #signal is stereo so taking second values
	print("            Pre emphasises Done")
	framed_data=framing(pre_emphasized_data,0.025,0.001,frequency)
	print("            Framing Done")
	windowed_data=windowing(framed_data)
	print("            Windowing Done")
	fft_power_spectrum_of_data=(fft_power_spectrum(windowed_data,512))
	print("            FFT Done")
	dd=mfcc(triangular_filter_banks(frequency,40,512,fft_power_spectrum_of_data))
	print("            Triangular Filter Bank")
	# frame_index=0
	# for frame_data in dd:
	# 	data["sample"+str(index)+"frame"+str(frame_index)]=framed_data.mean();
	# 	frame_index=frame_index+1
	# # print(str(parsed_audio_Data[index].emotion_category)+"    "+str(dd[0].mean()))
	index_of_mongo=np.array_str(np.arange(len(dd[1,:])))
	# print(np.append(index,np.mean(dd,1),axis=1)[1:5])
	data.update(dict(zip(index_of_mongo,np.mean(dd,1))))
	index=index+1
	emotion_database.data.insert_one(data)
	print(index)



# PLT.pcolormesh(mfcc(dd))
# # PLT.plot(dd[1])
