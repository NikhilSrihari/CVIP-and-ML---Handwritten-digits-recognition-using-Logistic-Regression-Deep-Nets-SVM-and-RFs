# For training, validation and testing
def loadMNISTData():
	import pickle
	import gzip
	filename = 'mnist.pkl.gz'
	f = gzip.open(filename, 'rb')
	training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
	f.close()
	return training_data, validation_data, test_data


# Only for testing
def loadUSPSData():
	from PIL import Image
	import os
	import numpy as np
	USPSMat  = []
	USPSTar  = []
	curPath  = 'USPSdata/Numerals'
	savedImg = []
	for j in range(0,10):
	    curFolderPath = curPath + '/' + str(j)
	    imgs =  os.listdir(curFolderPath)
	    for img in imgs:
	        curImg = curFolderPath + '/' + img
	        if curImg[-3:] == 'png':
	            img = Image.open(curImg,'r')
	            img = img.resize((28, 28))
	            savedImg = img
	            imgdata = (255-np.array(img.getdata()))/255
	            USPSMat.append(imgdata)
	            USPSTar.append(j)
	return np.array(USPSMat), np.array(USPSTar)