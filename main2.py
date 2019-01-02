import pandas as pd
import numpy as np
from MNIST_USPS_Data_PreProcessing import loadMNISTData, loadUSPSData
from logisticRegression import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from metrics import accuracy_score, cross_entropy_error, root_mean_square_error, confusion_matrix
import matplotlib.pyplot as mplt


def oneOfKCoding(target):
	target1 = pd.Series(target)
	return np.array(pd.get_dummies(target1))


'''
	num of rec X num of classes
'''
def fetchMNISTData():
	training_data, validation_data, testing_data = loadMNISTData()
	return {
			"training_data": {
				"inputs": training_data[0],
				"targets": oneOfKCoding(training_data[1]),
				"rawtargets": training_data[1]
				},
			"validation_data": {
				"inputs": validation_data[0],
				"targets": oneOfKCoding(validation_data[1]),
				"rawtargets": validation_data[1]
				},
			"testing_data": {
				"inputs": testing_data[0],
				"targets": oneOfKCoding(testing_data[1]),
				"rawtargets": testing_data[1]
				}
		   }


'''
	num of rec X num of classes
'''
def fetchUSPSData():
	USPSMat, USPSTar = loadUSPSData()
	return {
			"testing_data": {
				"inputs": USPSMat,
				"targets": oneOfKCoding(USPSTar),
				"rawtargets": USPSTar
				}
		   }


def trainLogisticRegression(data):
	LRs = []
	hyperParams_set = [ {"Eta": 0.005, "epochs": 10, "batchSize": 100} ]
						#{"Eta": 0.005, "epochs": 20, "batchSize": 100} ]
						#{"Eta": 0.005, "epochs": 100, "batchSize": 100},
						#{"Eta": 0.005, "epochs": 200, "batchSize": 100} ]
						#{"Eta": 0.005, "epochs": 1, "batchSize": 100},
						#{"Eta": 0.005, "epochs": 2, "batchSize": 100},
						#{"Eta": 0.01, "epochs": 1, "batchSize": 100},
						#{"Eta": 0.01, "epochs": 1, "batchSize": 100},
						#{"Eta": 0.01, "epochs": 1, "batchSize": 100},
						#{"Eta": 0.01, "epochs": 1, "batchSize": 100},
						#{"Eta": 0.01, "epochs": 2, "batchSize": 100} ]
	i=0
	for hyperParams in hyperParams_set:
		#try:
			print("	Now in : "+str(hyperParams))
			lr = LogisticRegression()
			lr.fit( np.transpose(data["training_data"]["inputs"]), np.transpose(data["training_data"]["targets"]), hyperParams["Eta"], hyperParams["epochs"], hyperParams["batchSize"])
			fig, ax = mplt.subplots()
			ax.plot(lr.training_error)
			fig.savefig("./graphs/LR_training_error_"+str(i)+'.jpg', dpi=fig.dpi)
			fig, ax = mplt.subplots()
			ax.plot(lr.training_accuracy)
			fig.savefig("./graphs/LR_training_accuracy_"+str(i)+'.jpg', dpi=fig.dpi)
			i=i+1
			training_y = np.transpose( lr.predict( np.transpose(data["training_data"]["inputs"]) ) )
			validation_y =  np.transpose( lr.predict( np.transpose(data["validation_data"]["inputs"]) ) )
			training_yF = np.transpose( lr.classificationPredict( np.transpose(training_y) ) )
			validation_yF = np.transpose( lr.classificationPredict( np.transpose(validation_y) ) )
			training_error = cross_entropy_error( data["training_data"]["targets"], training_y )
			validation_error = cross_entropy_error( data["validation_data"]["targets"], validation_y )
			training_accuracyScore = accuracy_score( data["training_data"]["targets"], training_yF )
			validation_accuracyScore = accuracy_score( data["validation_data"]["targets"], validation_yF )
			LRs.append( {	"lr": lr, "training_error": training_error, "validation_error": validation_error, "testingMNISTData_error": None, "testingUSPSData_error": None,
							"hyperParams": hyperParams, "training_accuracyScore": training_accuracyScore, 
							"validation_accuracyScore": validation_accuracyScore, "testingMNISTData_accuracyScore": None, "testingUSPSData_accuracyScore": None,
							"testingMNISTData_confusionMatrix": None, "testingUSPSData_confusionMatrix": None 	} )
			print("		training_error :  "+str(training_error))
			print("		training_accuracyScore :  "+str(training_accuracyScore))
			print("		validation_error :  "+str(validation_error))
			print("		validation_accuracyScore :  "+str(validation_accuracyScore))
		#except Exception as e:
		#	print("	Exception - "+str(e)+"- occured for : "+str(hyperParams))
	finalLR = min(LRs, key = lambda x: x["validation_error"])
	return finalLR


def testLogisticRegression(finalLRObject, data, dataType):
	finalLR = finalLRObject["lr"]
	testing_y = np.transpose( finalLR.predict( np.transpose(data["testing_data"]["inputs"]) ) )
	testing_yF = np.transpose( finalLR.classificationPredict( np.transpose(testing_y) ) )
	finalLRObject["testing"+dataType+"_error"] = cross_entropy_error( data["testing_data"]["targets"], testing_y )
	finalLRObject["testing"+dataType+"_confusionMatrix"], finalLRObject["testing"+dataType+"_accuracyScore"] = confusion_matrix( data["testing_data"]["targets"], testing_yF )
	return testing_y


def trainSVC(data):
	SVCs = []
	hyperParams_set = [ {"kernel": "linear"},
						{"kernel": "rbf", "gamma": 1},
						{"kernel": "rbf"},
						{"kernel": "poly"},
						{"kernel": "poly", "gamma": 1},
						{"kernel": "poly", "degree": 5},
						{"kernel": "sigmoid"},
						{"kernel": "sigmoid", "gamma": 1} ]
	for hyperParams in hyperParams_set:
		try:
			print("	Now in : "+str(hyperParams))
			if "gamma" in hyperParams:
				svc = SVC(kernel=hyperParams["kernel"], gamma=hyperParams["gamma"])
			elif "degree" in hyperParams:
				svc = SVC(kernel=hyperParams["kernel"], degree=hyperParams["degree"])
			else:
				svc = SVC(kernel=hyperParams["kernel"])
			svc.fit( data["training_data"]["inputs"][:500], data["training_data"]["rawtargets"][:500] )
			training_y = svc.predict( data["training_data"]["inputs"] )
			validation_y =  svc.predict( data["validation_data"]["inputs"] )
			training_error = root_mean_square_error( data["training_data"]["rawtargets"], training_y )
			validation_error = root_mean_square_error( data["validation_data"]["rawtargets"], validation_y )
			training_accuracyScore = accuracy_score( data["training_data"]["rawtargets"], training_y )
			validation_accuracyScore = accuracy_score( data["validation_data"]["rawtargets"], validation_y )
			SVCs.append( {	"svc": svc, "training_error": training_error, "validation_error": validation_error, "testingMNISTData_error": None, "testingUSPSData_error": None,
							"hyperParams": hyperParams,  "training_accuracyScore": training_accuracyScore, "validation_accuracyScore": validation_accuracyScore, 
							"testingMNISTData_accuracyScore": None, "testingUSPSData_accuracyScore": None,
							"testingMNISTData_confusionMatrix": None, "testingUSPSData_confusionMatrix": None 	} )
			print("		training_error :  "+str(training_error))
			print("		training_accuracyScore :  "+str(training_accuracyScore))
			print("		validation_error :  "+str(validation_error))
			print("		validation_accuracyScore :  "+str(validation_accuracyScore))
		except Exception as e:
			print("	Exception - "+str(e)+"- occured for : "+str(hyperParams))
	finalSVC = min(SVCs, key = lambda x: x["validation_error"])
	return finalSVC


def testSVC(finalSVCObject, data, dataType):
	finalSVC = finalSVCObject["svc"]
	testing_y = oneOfKCoding( finalSVC.predict( data["testing_data"]["inputs"] ) )
	finalSVCObject["testing"+dataType+"_error"] = root_mean_square_error( data["testing_data"]["targets"], testing_y )
	finalSVCObject["testing"+dataType+"_confusionMatrix"], finalSVCObject["testing"+dataType+"_accuracyScore"] = confusion_matrix( data["testing_data"]["targets"], testing_y )
	return testing_y


def trainRFC(data):
	RFCs = []
	hyperParams_set = [ {"n_estimators": 5, "criterion": "gini"}]
	''',
						{"n_estimators": 10, "criterion": "gini"},
						{"n_estimators": 15, "criterion": "gini"},
						{"n_estimators": 20, "criterion": "gini"},
						{"n_estimators": 25, "criterion": "gini"},
						{"n_estimators": 30, "criterion": "gini"},
						{"n_estimators": 50, "criterion": "gini"},
						{"n_estimators": 5, "criterion": "entropy"},
						{"n_estimators": 10, "criterion": "entropy"},
						{"n_estimators": 15, "criterion": "entropy"},
						{"n_estimators": 20, "criterion": "entropy"},
						{"n_estimators": 25, "criterion": "entropy"},
						{"n_estimators": 30, "criterion": "entropy"}, 
						{"n_estimators": 50, "criterion": "entropy"} ]'''
	for hyperParams in hyperParams_set:
		try:
			print("	Now in : "+str(hyperParams))
			rfc = RandomForestClassifier(n_estimators=hyperParams["n_estimators"], criterion=hyperParams["criterion"])
			rfc.fit( data["training_data"]["inputs"], data["training_data"]["targets"] )
			training_y = rfc.predict( data["training_data"]["inputs"] ) 
			validation_y =  rfc.predict( data["validation_data"]["inputs"] ) 
			training_error = root_mean_square_error( data["training_data"]["targets"], training_y )
			validation_error = root_mean_square_error( data["validation_data"]["targets"], validation_y )
			training_accuracyScore = accuracy_score( data["training_data"]["targets"], training_y )
			validation_accuracyScore = accuracy_score( data["validation_data"]["targets"], validation_y )
			RFCs.append( {	"rfc": rfc, "training_error": training_error, "validation_error": validation_error, "testingMNISTData_error": None, "testingUSPSData_error": None,
							"hyperParams": hyperParams,  "training_accuracyScore": training_accuracyScore, "validation_accuracyScore": validation_accuracyScore, 
							"testingMNISTData_accuracyScore": None, "testingUSPSData_accuracyScore": None,
							"testingMNISTData_confusionMatrix": None, "testingUSPSData_confusionMatrix": None 	} )
			print("		training_error :  "+str(training_error))
			print("		training_accuracyScore :  "+str(training_accuracyScore))
			print("		validation_error :  "+str(validation_error))
			print("		validation_accuracyScore :  "+str(validation_accuracyScore))
		except Exception as e:
			print("	Exception - "+str(e)+"- occured for : "+str(hyperParams))
	finalRFC = min(RFCs, key = lambda x: x["validation_error"])
	return finalRFC


def testRFC(finalRFCObject, data, dataType):
	finalRFC = finalRFCObject["rfc"]
	testing_y = finalRFC.predict( data["testing_data"]["inputs"] )
	finalRFCObject["testing"+dataType+"_error"] = root_mean_square_error( data["testing_data"]["targets"], testing_y )
	finalRFCObject["testing"+dataType+"_confusionMatrix"], finalRFCObject["testing"+dataType+"_accuracyScore"] = confusion_matrix( data["testing_data"]["targets"], testing_y )
	return testing_y


def trainNN(data):
	NNs = []
	RFCs = []
	input_size = len(data["training_data"]["inputs"][0])	
	output_size = len(data["training_data"]["targets"][0])
	hyperParams_set = [ #{"NN_type":"DNN","layers":[{"numOfNodes":500,"actFcn":"relu","dropout":0.1},{"numOfNodes":200,"actFcn":"relu","dropout":0.1},{"actFcn":"softmax"}],"optimizerParams":{"type":"SGD","lr":0.1,"decay":1e-6,"momentum":0.3},"lossFcn":"mean_squared_error","numOfEpochs":50,"modelBatchSize":128,"tbBatchSize":32,"earlyPatience":100},
						#{"NN_type":"DNN","layers":[{"numOfNodes":500,"actFcn":"relu","dropout":0.1},{"numOfNodes":500,"actFcn":"relu","dropout":0.1},{"actFcn":"softmax"}],"optimizerParams":{"type":"SGD","lr":0.1,"decay":1e-6,"momentum":0.3},"lossFcn":"mean_squared_error","numOfEpochs":50,"modelBatchSize":128,"tbBatchSize":32,"earlyPatience":100},
						#{"NN_type":"DNN","layers":[{"numOfNodes":500,"actFcn":"relu","dropout":0.1},{"numOfNodes":1000,"actFcn":"relu","dropout":0.1},{"actFcn":"softmax"}],"optimizerParams":{"type":"SGD","lr":0.1,"decay":1e-6,"momentum":0.3},"lossFcn":"mean_squared_error","numOfEpochs":50,"modelBatchSize":128,"tbBatchSize":32,"earlyPatience":100},
						#{"NN_type":"DNN","layers":[{"numOfNodes":500,"actFcn":"relu","dropout":0.1},{"numOfNodes":500,"actFcn":"relu","dropout":0.1},{"numOfNodes":500,"actFcn":"relu","dropout":0.1},{"actFcn":"softmax"}],"optimizerParams":{"type":"SGD","lr":0.1,"decay":1e-6,"momentum":0.3},"lossFcn":"mean_squared_error","numOfEpochs":50,"modelBatchSize":128,"tbBatchSize":32,"earlyPatience":100},

						{"NN_type":"CNN","layers":[{"type":"CNN","numOfFilters":32,"filterDim":3,"padding":"same","actFcn":"relu"},{"type":"POOL","pool_size":2,"dropout":0.25},{"type":"CNN","numOfFilters":64,"filterDim":3,"padding":"same","actFcn":"relu"},{"type":"CNN","numOfFilters":64,"filterDim":3,"padding":"same","actFcn":"relu"},{"type":"POOL","pool_size":2,"dropout":0.25},{"type":"CNN","numOfFilters":128,"filterDim":3,"padding":"same","actFcn":"relu"},{"type":"CNN","numOfFilters":128,"filterDim":3,"padding":"same","actFcn":"relu"},{"type":"CNN","numOfFilters":128,"filterDim":3,"padding":"same","actFcn":"relu"},{"type":"POOL","pool_size":2,"dropout":0.25},{"type":"FLATTEN"},{"type":"DENSE","numOfNodes":512,"actFcn":"relu"},{"type":"BATCHNORMALIZATION_POSTFIRSTDENSE","dropout":0.5},{"type":"DENSE","numOfNodes":10,"actFcn":"softmax"}],"optimizerParams":{"type":"SGD","lr":0.1,"decay":1e-6,"momentum":0.3},"lossFcn":"categorical_crossentropy","numOfEpochs":50,"modelBatchSize":128,"tbBatchSize":32,"earlyPatience":None} ]
	i=0
	for hyperParams in hyperParams_set:
		#try:
		print("	Now in : "+str(hyperParams))
		NN = NeuralNetwork(NN_type=hyperParams["NN_type"], input_size=input_size, output_size=output_size, layers=hyperParams["layers"], optimizerParams=hyperParams["optimizerParams"], lossFcn=hyperParams["lossFcn"])
		history = NN.fit(data["training_data"]["inputs"], data["training_data"]["targets"], data["validation_data"]["inputs"], data["validation_data"]["targets"], hyperParams["numOfEpochs"], hyperParams["modelBatchSize"], hyperParams["tbBatchSize"], hyperParams["earlyPatience"])
		fig, ax = mplt.subplots()
		ax.plot(history.history['loss'])
		fig.savefig("./graphs/NN_loss_"+str(i)+'.jpg', dpi=fig.dpi)
		fig, ax = mplt.subplots()
		ax.plot(history.history['val_loss'])
		fig.savefig("./graphs/NN_val_loss_"+str(i)+'.jpg', dpi=fig.dpi)
		fig, ax = mplt.subplots()
		ax.plot(history.history['acc'])
		fig.savefig("./graphs/NN_acc_"+str(i)+'.jpg', dpi=fig.dpi)
		fig, ax = mplt.subplots()
		ax.plot(history.history['val_acc'])
		fig.savefig("./graphs/NN_val_acc_"+str(i)+'.jpg', dpi=fig.dpi)
		i=i+1
		training_error = history.history['loss'][len(history.history['loss'])-1]
		validation_error = history.history['val_loss'][len(history.history['val_loss'])-1]
		training_accuracyScore = history.history['acc'][len(history.history['acc'])-1] * 100
		validation_accuracyScore = history.history['val_acc'][len(history.history['val_acc'])-1] * 100
		NNs.append( {   "NN":NN, "training_error":training_error, "validation_error":validation_error, "testingMNISTData_error": None, "testingUSPSData_error": None, 
						"hyperParams": hyperParams, "training_accuracyScore": training_accuracyScore, "validation_accuracyScore": validation_accuracyScore,
						"testingMNISTData_accuracyScore": None, "testingUSPSData_accuracyScore": None,
						"testingMNISTData_confusionMatrix": None, "testingUSPSData_confusionMatrix": None 	} )
		print("		training_error :  "+str(training_error))
		print("		training_accuracyScore :  "+str(training_accuracyScore))
		print("		validation_error :  "+str(validation_error))
		print("		validation_accuracyScore :  "+str(validation_accuracyScore))
		#except Exception as e:
			#print("	Exception - "+str(e)+"- occured for : "+str(hyperParams))
	finalNN = min(NNs, key = lambda x: x["validation_error"])
	return finalNN


def testNN(finalNNObject, data, dataType):
	finalNN = finalNNObject["NN"]
	testing_y = finalNN.predict( data["testing_data"]["inputs"] )
	finalNNObject["testing"+dataType+"_error"] = root_mean_square_error( data["testing_data"]["targets"], testing_y )
	finalNNObject["testing"+dataType+"_confusionMatrix"], finalNNObject["testing"+dataType+"_accuracyScore"] = confusion_matrix( data["testing_data"]["targets"], testing_y )
	return testing_y


def createMetaClassifierObject(technique):
	return { "technique": technique, "testingMNISTData_accuracyScore": None, "testingUSPSData_accuracyScore": None, "testingMNISTData_confusionMatrix": None, "testingUSPSData_confusionMatrix": None   }


def combineAllClassifierResults(technique, T1, T2, T3, T4):
	y = []
	if (technique=="MAX_VOTING"):
		for t1, t2, t3, t4 in zip(T1, T2, T3, T4):
			temp = [ np.argmax(t1), np.argmax(t2), np.argmax(t3), np.argmax(t4) ]
			m = stats.mode(np.array(temp))
			y.append( m[0][0] )
	return np.array(y)


def testMetaClassifier(finalMetaClassifierObject, T1, T2, T3, T4, data, dataType):
	testing_y = combineAllClassifierResults( finalMetaClassifierObject["technique"], T1, T2, T3, T4 )
	finalMetaClassifierObject["testing"+dataType+"_confusionMatrix"], finalNNObject["testing"+dataType+"_accuracyScore"] = confusion_matrix( data["testing_data"]["rawtargets"], testing_y )
	

def main():
	print("Starting Data Read: ")
	MNISTData = fetchMNISTData()
	print("	MNISTData Done.")
	
	print("	USPSData Done.")
	print("Data Read Complete.")
	print()
	print("Starting Logistic Regression: ")
	finalLRObject = trainLogisticRegression(MNISTData)
	del MNISTData
	'''
	USPSData = fetchUSPSData()
	LR_USPSData_testing_y = testLogisticRegression(finalLRObject, USPSData, "USPSData")
	print("	FinalLRObject = "+str(finalLRObject))
	print("Logistic Regression Complete.")
	print()
	'''
	'''print("Starting SVC: ")
	finalSVCObject = trainSVC(MNISTData)
	SVC_MNISTData_testing_y = testSVC(finalSVCObject, MNISTData, "MNISTData")
	SVC_USPSData_testing_y = testSVC(finalSVCObject, USPSData, "USPSData")
	print("	FinalSVCObject = "+str(finalSVCObject))
	print("SVC Complete.")
	print()
	print("Starting RFC: ")
	finalRFCObject = trainRFC(MNISTData)
	RFC_MNISTData_testing_y = testRFC(finalRFCObject, MNISTData, "MNISTData")
	RFC_USPSData_testing_y = testRFC(finalRFCObject, USPSData, "USPSData")
	print("	FinalRFCObject = "+str(finalRFCObject))
	print("RFC Complete.")
	print()
	print("Starting NN: ")
	finalNNObject = trainNN(MNISTData)
	NN_MNISTData_testing_y = testNN(finalNNObject, MNISTData, "MNISTData")
	NN_USPSData_testing_y = testNN(finalNNObject, USPSData, "USPSData")
	print("	FinalNNObject = "+str(finalNNObject))
	print("NN Complete.")
	print()
	print("Starting MetaClassifier: ")
	finalMetaClassifierObject = createMetaClassifierObject("MAX_VOTING")
	testMetaClassifier(finalMetaClassifierObject, LR_MNISTData_testing_y, SVC_MNISTData_testing_y, RFC_MNISTData_testing_y, NN_MNISTData_testing_y, MNISTData, "MNISTData")
	testMetaClassifier(finalMetaClassifierObject, LR_USPSData_testing_y, SVC_USPSData_testing_y, RFC_USPSData_testing_y, NN_USPSData_testing_y, USPSData, "USPSData")
	print("	FinalMetaClassifierObject = "+str(finalMetaClassifierObject))
	print("MetaClassifier Complete.")
	print()'''


main()