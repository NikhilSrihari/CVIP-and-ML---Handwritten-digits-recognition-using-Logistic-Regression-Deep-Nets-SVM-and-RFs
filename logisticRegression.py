import pandas as pd
import numpy as np
import math


class LogisticRegression:


	def __init__(self):
		self.M = None
		self.K = None
		self.W = None
		self.b = None
		

	def fit(self, X0, T0, Eta, numberOfIterations):
		X = X0.copy()
		T = T0.copy()
		self.M = len(X)
		self.K = len(T)
		N = len(X[0])
		self.training_error = []
		self.training_accuracy = []
		b = np.zeros(self.K)
		W_current = np.zeros((self.M, self.K))
		for epoch in range(0, epochs):
			for Iter in range(0, N):
				x = X[:,i]
				t = T[:,i]
				a = np.dot((np.transpose(W_current)), x) + b
				a_exp = np.exp(a)
				a_exp_Sum = ((np.transpose(a_exp)).sum(axis=0))
				y = a_exp / a_exp_Sum
				diff = (y - t).reshape((1, self.K))
				deltaE = np.dot( x.reshape((self.M, 1)), diff )
				deltaW = -1 * Eta * deltaE
				W_current = W_current + deltaW
		self.W = W_current
		self.b = b


	def predict(self, X0):
		X = X0.copy()
		B = np.transpose(np.tile(self.b, (len(X[0]),1)))
		A = np.dot((np.transpose(self.W)), X) + B
		A_exp = np.exp(A)
		A_exp_Sum = ((np.transpose(A_exp)).sum(axis=1))
		Y = A_exp / A_exp_Sum
		return Y


	def classificationPredict(self, Y0):
		Y = Y0.copy()
		Y_ = np.zeros((len(Y), len(Y[0])))
		i=0
		while(i<len(Y[0])):
			Y_[(np.argmax(Y[:,i]))][i] = 1.0
			i=i+1
		return Y_