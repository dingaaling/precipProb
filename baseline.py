import os
import numpy as np
import pandas as pd

# Load Test Data
def loadData(dataPath):

    for i, csvFile in enumerate(os.listdir(dataPath)):

        print(i)
        filePath = os.path.join(dataPath, str(csvFile))

        if i == 8000:
            testData = pd.read_csv(filePath)
        elif (i > 8000 and i < 8500):
            csvData = pd.read_csv(filePath)
            testData = pd.concat([testData, csvData])

    testData.iloc[:,1:].to_csv("testData.csv")

# Split array into precipProb (feature) and didPrecip (label)
def dataSplit(testFile):
    testProbs = testFile.iloc[:,4]
    testLabels = testFile.iloc[:,7]
    testLabels = pd.to_numeric(testLabels, errors='coerce', downcast='float')

    return testProbs, testLabels

#Calculate the max precipProb for given list of hourly probabilities
def maxProbability(probList):
    return np.max(probList)

#Calculate the joint precipProb (assuming independence) for list of hourly probabilities
def jointProbability(probList):
    inv = 1-(probList)
    noRainProb = inv.cumprod()
    return((1-noRainProb.iloc[-1]))

def avgProbability(probList):
    return np.mean(probList)

#Generate group didPrecip label for list of hours
def groupPrecipLabel(testLabels, n):

    groundTruth = np.zeros(len(testLabels))

    for i in range(0, len(testLabels), n):
        subsetLabels = testLabels.iloc[i:i+n]

        if (subsetLabels.cumsum().iloc[-1] > 0) :
            groundTruth[i] = 1
        else:
            groundTruth[i] = 0

    return groundTruth

#For a given "n" number of hours, calculate the max and joint probabilities
def probCalc(testProbs, n):

    maxProb = np.zeros(len(testProbs))
    jointProb = np.zeros(len(testProbs))
    avgProb = np.zeros(len(testProbs))

    for i in range(0, len(testProbs), n):
        subsetProbs = testProbs.iloc[i:i+n]
        maxProb[i] = maxProbability(subsetProbs)
        jointProb[i] = jointProbability(subsetProbs)
        avgProb[i] = avgProbability(subsetProbs)

    return maxProb, jointProb, avgProb

#Quantify Mean Squared Error against actual didPrecip for that time range
def errorCalc(groundTruth, predProb):
    mse = np.square(groundTruth - predProb).mean()
    ssd = np.sum(np.square(groundTruth - predProb))
    return(mse, ssd)


if __name__ == '__main__':

    testData = pd.read_csv("testData.csv")
    testProbs, testLabels = dataSplit(testData)
    maxProb, jointProb, avgProb = probCalc(testProbs, 24)
    groundTruth = groupPrecipLabel(testLabels, 24)

    print("Max Probability Method")
    mse, ssd = errorCalc(groundTruth, maxProb)
    print("Mean Squared Error: ", mse)
    print("Sum of Squared Differences: ", ssd)

    print("\nJoint Probability Method")
    mse, ssd = errorCalc(groundTruth, jointProb)
    print("Mean Squared Error: ", mse)
    print("Sum of Squared Differences: ", ssd)
