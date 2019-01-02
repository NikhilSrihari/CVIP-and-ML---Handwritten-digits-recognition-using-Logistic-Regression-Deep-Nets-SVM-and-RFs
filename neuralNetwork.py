import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping, TensorBoard
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
import math
import matplotlib.pyplot as mplt


class NeuralNetwork():

    def __init__(self, NN_type, input_size, output_size, layers, optimizerParams, lossFcn):
        self.NN_type = NN_type
        self.NN = Sequential()
        if(self.NN_type=="DNN"):
            numOfLayers = len(layers)
            (self.NN).add(Dense(layers[0]["numOfNodes"], input_dim=input_size))
            (self.NN).add(Activation(layers[0]["actFcn"]))
            (self.NN).add(Dropout(layers[0]["dropout"]))
            for layer in layers[1:(numOfLayers-1)]:
                (self.NN).add(Dense(layer["numOfNodes"]))
                (self.NN).add(Activation(layer["actFcn"]))
                (self.NN).add(Dropout(layer["dropout"]))
            (self.NN).add(Dense(output_size))
            (self.NN).add(Activation(layers[numOfLayers-1]["actFcn"]))
            if(optimizerParams["type"]=="SGD"):
                optimizer = optimizers.SGD( lr=optimizerParams["lr"], decay=optimizerParams["decay"], momentum=optimizerParams["momentum"], nesterov=True)
            (self.NN).compile(optimizer=optimizer, loss=lossFcn, metrics=['accuracy'])
        else:
            inputShape = ((int)(math.sqrt(input_size)), (int)(math.sqrt(input_size)), 1)
            chanDim = 1
            # (CONV => RELU) * 1 => POOL layer set
            # (CONV => RELU) * 2 => POOL layer set
            # (CONV => RELU) * 3 => POOL layer set
            # first (and only) set of FC => RELU layers
            # softmax classifier
            (self.NN).add(Conv2D(layers[0]["numOfFilters"], (layers[0]["filterDim"], layers[0]["filterDim"]), padding=layers[0]["padding"], input_shape=inputShape))
            (self.NN).add(Activation(layers[0]["actFcn"]))
            (self.NN).add(BatchNormalization(axis=chanDim))
            for layer in layers[1:]:
                if (layer["type"]=="POOL"):
                    (self.NN).add(MaxPooling2D(pool_size=(layer["pool_size"], layer["pool_size"])))
                    (self.NN).add(Dropout(layer["dropout"]))
                elif (layer["type"]=="CONV"):
                    (self.NN).add(Conv2D(layer["numOfFilters"], (layer["filterDim"], layer["filterDim"]), padding=layer["padding"]))
                    (self.NN).add(Activation(layer["actFcn"]))
                    (self.NN).add(BatchNormalization(axis=chanDim))
                elif (layer["type"]=="FLATTEN"):
                    (self.NN).add(Flatten())
                elif (layer["type"]=="DENSE"):
                    print("sa"+(str)(layer["numOfNodes"]))
                    (self.NN).add(Dense(layer["numOfNodes"]))
                    (self.NN).add(Activation(layer["actFcn"]))
                elif (layer["type"]=="BATCHNORMALIZATION_POSTFIRSTDENSE"):
                    (self.NN).add(BatchNormalization())
                    (self.NN).add(Dropout(layer["dropout"]))
            if(optimizerParams["type"]=="SGD"):
                optimizer = optimizers.SGD( lr=optimizerParams["lr"], decay=optimizerParams["decay"], momentum=optimizerParams["momentum"], nesterov=True)
            (self.NN).compile(optimizer=optimizer, loss=lossFcn, metrics=['accuracy'])


    def __createImages(self, concatenatedImgs):
        input_size = len(concatenatedImgs[0])
        op = []
        for concatenatedImg in concatenatedImgs:
            op.append( np.resize(concatenatedImg, ( (int)(math.sqrt(input_size)), (int)(math.sqrt(input_size)), 1 )) )
        return np.array(op)


    def fit(self, X_t, T_t, X_v, T_v, numOfEpochs, modelBatchSize, tbBatchSize, earlyPatience=None):
        if(self.NN_type=="DNN"):
            history = (self.NN).fit( X_t, T_t, validation_data=(X_v, T_v), epochs=numOfEpochs, batch_size=modelBatchSize, verbose=0,
                                 callbacks = [TensorBoard(log_dir='logs', batch_size= tbBatchSize, write_graph= True), EarlyStopping(monitor='val_loss', verbose=0, patience=earlyPatience, mode='min')])
        else:
            history = (self.NN).fit( self.__createImages(X_t), T_t, validation_data=(self.__createImages(X_v), T_v), epochs=numOfEpochs, batch_size=modelBatchSize)
        return history


    def predict(self, X):
        if(self.NN_type=="DNN"):
            return (self.NN).predict(X)
        else:
            return (self.NN).predict(self.__createImages(X), batch_size=32)