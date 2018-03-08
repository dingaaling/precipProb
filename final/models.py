# Jennifer Ding
# Python 3 - Supervised and statistical models for predicting probability

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

def supervised(trainFeatures, trainLabels, testFeatures, testLabels, printScore):
    # Fit supervised learning methods
    clf, mlp, rf = LogisticRegression(), MLPClassifier(), RandomForestClassifier()
    clf.fit(trainFeatures, trainLabels)
    mlp.fit(trainFeatures, trainLabels)
    rf.fit(trainFeatures, trainLabels)

    # Run predictions and quantify error
    pred_lr, pred_mlp, pred_rf = clf.predict(testFeatures), mlp.predict(testFeatures), rf.predict(testFeatures)
    prob_lr, prob_mlp, prob_rf = clf.predict_proba(testFeatures), mlp.predict_proba(testFeatures), rf.predict_proba(testFeatures)

    if printScore==True:
        print("Testing Scores:")
        print("Logistic Regression Score: ", clf.score(testFeatures, testLabels), "MSE: ", mean_squared_error(testLabels, pred_lr))
        print("MLP Score: ", mlp.score(testFeatures, testLabels), "MSE: ", mean_squared_error(testLabels, pred_mlp))
        print("Random Forest Score: ", rf.score(testFeatures, testLabels), "MSE: ", mean_squared_error(testLabels, pred_rf))

    return pred_lr, pred_mlp, pred_rf

#Calculate the max precipProb for given list of hourly probabilities
def maxProbability(probList):
    return np.max(probList)

#Calculate the joint precipProb (assuming independence) for list of hourly probabilities
def jointProbability(probList):
    inv = 1-(probList)
    noRainProb = inv.cumprod()
    return((1-noRainProb[-1]))

def avgProbability(probList):
    return np.mean(probList)

def baseline(testProbs, testLabels, n, printScore):

    maxProb = np.zeros(int(len(testProbs)/n))
    jointProb = np.zeros(int(len(testProbs)/n))
    avgProb = np.zeros(int(len(testProbs)/n))

    for i in range(0, len(testProbs), n):
        if (i+n < len(testProbs)):
            ind = int(i/n)
            subsetProbs = testProbs[i:i+n]
            maxProb[ind] = maxProbability(subsetProbs)
            jointProb[ind] = jointProbability(subsetProbs)
            avgProb[ind] = avgProbability(subsetProbs)

    if printScore==True:
        print("Testing Scores:")
        print("Max Probability MSE", mean_squared_error(testLabels, maxProb))
        print("Joint Probability MSE", mean_squared_error(testLabels, jointProb))
        print("Average Probability MSE", mean_squared_error(testLabels, avgProb))

    return maxProb, jointProb, avgProb
