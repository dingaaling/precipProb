# Jennifer Ding
# Python 3 - Predict probability of precipitation in a given time range
# Input - n x 7 feature vector for n # of hours
# Output - precipitation probability prediction & visualizations of error

import os, sys
import numpy as np
import pandas as pd
from dataTransforms import *
from models import *
from sklearn.metrics import mean_squared_error

def runModel(trainFeatures, trainLabels, testFeatures, testLabels, trainFeaturesProb, testFeaturesProb, featureType, n):

    # using all 7 weather features
    if featureType == "allFeatures":
        trainGroupFeatures, trainGroupLabels = groupFeatures(trainFeatures, n), groupLabels(trainLabels, n)
        testGroupFeatures, testGroupLabels = groupFeatures(testFeatures, n), groupLabels(testLabels, n)

        print("\nSupervised Model with feature type %s and for %d hour groups" % (featureType, n))
        pred_lr, pred_mlp, pred_rf = supervised(trainGroupFeatures, trainGroupLabels, testGroupFeatures, testGroupLabels, printScore=True)

        finalPred = np.mean([pred_lr, pred_mlp, pred_rf], axis=0)
        print("Final Pred MSE:", mean_squared_error(testGroupLabels, finalPred))

        return finalPred

    # using distribution of hourly precipitation probability features
    elif featureType == "probDistrFeatures":
        trainGroupFeatures, trainGroupLabels = groupProbFeatures(trainFeaturesProb, n), groupLabels(trainLabels, n)
        testGroupFeatures, testGroupLabels = groupProbFeatures(testFeaturesProb, n), groupLabels(testLabels, n)

        print("\nSupervised Model with feature type %s and for %d hour groups" % (featureType, n))
        pred_lr, pred_mlp, pred_rf = supervised(trainGroupFeatures, trainGroupLabels, testGroupFeatures, testGroupLabels, printScore=True)

        finalPred = np.mean([pred_lr, pred_mlp, pred_rf], axis=0)
        print("Final Pred MSE:", mean_squared_error(testGroupLabels, finalPred))

        return finalPred

    # using list of probabilities for a given time range
    elif featureType == "probList":
        testGroupLabels = groupLabels(testLabels, n)

        print("\nBaseline Model with feature type %s and for %d hour groups" % (featureType, n))
        pred_max, pred_joint, pred_avg = baseline(testFeaturesProb, testGroupLabels, n, printScore=True)

        finalPred = np.mean([pred_max, pred_joint, pred_avg], axis=0)
        print("Final Pred MSE:", mean_squared_error(testGroupLabels, finalPred))

        return finalPred

if __name__ == '__main__':

    # parse input files
    args = sys.argv

    # check if an input file is specified
    if len(args) > 1:
        fileName = str(args[1])

        inputVal = []
        for line in open(fileName):
            line = line.strip().split(' ')
            inputVal.append(line)

        if len(inputVal)!=3:
            print("\nPlease reformat input file to line 1: testData filename, line 2: number of hourly forecasts in prediction, line 3: feature type")
        else:
            testFile, n, featureType = ''.join(inputVal[0]), int(''.join(inputVal[1])), ''.join(inputVal[2])
            print("\nCalculating precipitation probability for data from: %s, in n = %d hour groups, and %s feature type" % (testFile, n, featureType))
    # use default parameters if input file is not used
    else:

        testFile, n, featureType = "testData.csv", 24, "allFeatures"
        print("Input file not specified. Default parameters used")
        print("\nCalculating precipitation probability for data from: %s, in n = %d hour groups, and %s feature type" % (testFile, n, featureType))

    # Load data, clean, and normalize - trainData (8000 24 hour x 7 day samples), testData default (6,000), valData (2,144)
    # Input data must be of shape n x 8 (7 features and 1 didPrecip feature)
    trainData = pd.read_csv("trainData.csv")
    testData = pd.read_csv(testFile)

    # Extract full flattened feature and label arrays
    trainFeatures, trainLabels, testFeatures, testLabels, \
    trainFeaturesProb, testFeaturesProb = prepData(trainData, testData)

    # Run model to attain precipitation probability
    precipProbPred = runModel(trainFeatures, trainLabels, testFeatures, testLabels, trainFeaturesProb, testFeaturesProb, featureType, n)
    print("Predicted Precipitation Probabilities:", precipProbPred)

    # For Testing Results to quantify results of different models, n, and feature types (see: TestingResults.jpg)
    # test_n = [1, 12, 24, 48]
    # for n in test_n:
    #     pred = runModel(trainFeatures, trainLabels, testFeatures, testLabels, trainFeaturesProb, testFeaturesProb, "allFeatures", n)
    #     pred = runModel(trainFeatures, trainLabels, testFeatures, testLabels, trainFeaturesProb, testFeaturesProb, "probDistrFeatures", n)
    #     pred = runModel(trainFeatures, trainLabels, testFeatures, testLabels, trainFeaturesProb, testFeaturesProb, "probList", n)
