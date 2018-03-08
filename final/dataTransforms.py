# Jennifer Ding
# Python 3 - data preparation, cleaning, grouping techniques

import os
import numpy as np
import pandas as pd
from sklearn import preprocessing

def nulltozero(labels):
    for i, label in enumerate(labels):
        if label=="null":
            labels[i] = 0
    return labels

def prepData(trainData, testData):

    # Partition data into features (0-6) and didPrecip (7)
    trainFeatures, trainLabels = np.array(trainData.iloc[:,:7]).astype(float), np.array(trainData.iloc[:,7])
    testFeatures, testLabels = np.array(testData.iloc[:,:7]).astype(float), np.array(testData.iloc[:,7])
    trainFeaturesProb, testFeaturesProb = trainFeatures[:,4], testFeatures[:,4]

    # Scale non-probability features
    scaler = preprocessing.StandardScaler()
    trainFeatures, testFeatures, = scaler.fit_transform(trainFeatures), scaler.fit_transform(testFeatures)

    # Set null didPrecip values to zero
    trainLabels, testLabels = nulltozero(trainLabels).astype(int), nulltozero(testLabels).astype(int)

    return trainFeatures, trainLabels, testFeatures, testLabels, trainFeaturesProb, testFeaturesProb


def groupFeatures(features, n):

    featureSize = features.shape
    trainFeaturesGroup = np.zeros((int(featureSize[0]/n), featureSize[1]*n))

    for i in range(0, len(features), n):
        if (i+n < len(features)):
            ind = int(i/n)
            trainFeaturesGroup[ind] = np.concatenate(features[i:i+n])

    return trainFeaturesGroup

def groupLabels(labels, n):

    trainLabelsGroup = np.zeros(int(len(labels)/n))

    for i in range(0, len(labels), n):
        if (i+n < len(labels)):
            subsetLabels = labels[i:i+n]
            ind = int(i/n)
            if (sum(subsetLabels) > 0):
                trainLabelsGroup[ind] = 1
            else:
                trainLabelsGroup[ind] = 0

    return trainLabelsGroup


def groupProbFeatures(features, n):

    featureSize = features.shape
    trainFeaturesGroup = np.zeros((int(featureSize[0]/n), 101))
    groupInd = 0

    for i in range(0, len(features), n):
        if (i+n <= len(features)):
            subsetProbs = features[i:i+n]
            for prob in subsetProbs:
                probInd = int(prob*100)
                trainFeaturesGroup[groupInd, probInd] +=1
        groupInd+=1

    return trainFeaturesGroup
