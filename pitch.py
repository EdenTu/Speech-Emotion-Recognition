import temp
import numpy as np
index=0


def delta_calculator(mfccs_of_whole_signal,resolution_factor,frame_index):
	delta_mfcc=list()
	targer_frame=mfccs_of_whole_signal[frame_index]
	##calculating denominator
	denominator=0
	for resolution_index in range(resolution_factor):
		denominator+=(resolution_index+1)**2
	for index_of_ceptral_coeff in range(len(targer_frame)):
		numerator=0
		for resolution_index in range(resolution_factor):
			max_index=(resolution_index+1+frame_index)
			min_index=(frame_index-resolution_index-1)
			if(min_index<0):
				min_index=0
			if(max_index>=len(mfccs_of_whole_signal)):
				max_index=len(mfccs_of_whole_signal)-1
			numerator+=(resolution_index+1)*(mfccs_of_whole_signal[max_index][index_of_ceptral_coeff]-mfccs_of_whole_signal[min_index][index_of_ceptral_coeff])
		delta_mfcc.append(numerator/denominator)
	return delta_mfcc	
