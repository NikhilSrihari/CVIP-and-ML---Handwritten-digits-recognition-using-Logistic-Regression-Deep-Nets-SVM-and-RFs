import numpy as np
import math


def root_mean_square_error(idealValues, predictedValues):
	return math.sqrt( (np.sum(np.square(idealValues - predictedValues))) / (len(predictedValues)) )


def cross_entropy_error(idealValues, predictedValues):
	predictedValues_ln = np.log(predictedValues)
	N = len(predictedValues_ln)
	K = len(predictedValues_ln[0])
	sum = 0
	i=0
	while(i<N):
		e = 0
		j=0
		while(j<K):
			e = e + idealValues[i][j]*predictedValues_ln[i][j]
			j=j+1
		e = -1 * e
		sum = sum + e
		i=i+1
	return sum/N


def accuracy_score(idealValues, predictedValues):
	numOfEntries = len(idealValues)
	correctPredictionsCnt = 0
	i=0
	while(i<numOfEntries):
		if ( (idealValues.ndim==1 and idealValues[i]==predictedValues[i]) or (idealValues.ndim==2 and np.array_equal(idealValues[i],predictedValues[i])) ):
			correctPredictionsCnt=correctPredictionsCnt+1
		i=i+1
	return ((correctPredictionsCnt/numOfEntries)*100)


def confusion_matrix(idealValues, predictedValues):
	numOfEntries = len(idealValues)
	K = len(idealValues[0])
	C = np.zeros((K, K))
	i=0
	while(i<numOfEntries):
		if(idealValues.ndim==1):
			idealValues_class = idealValues[i]
			predictedValues_class = predictedValues[i]
		else:
			idealValues_class = np.argmax(idealValues[i])
			predictedValues_class = np.argmax(predictedValues[i])
		C[idealValues_class][predictedValues_class] = C[idealValues_class][predictedValues_class] + 1
		i=i+1
	s = 0
	k=0
	while(k<K):
		s = s + C[k][k]
		k=k+1
	return C, ((s/numOfEntries)*100)