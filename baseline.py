import os
import numpy as np
import pandas as pd

# Convert CSV indices to array
def loadIndData(indPath):
    indices = pd.read_csv(indPath, names=['index'], dtype=np.int32)
    return indices['index'].tolist()

def maxProbability(probList):
    return np.max(probList)

def jointProbability(probList):
    inv = 1-(probList)
    noRainProb = inv.cumprod()
    return((1-noRainProb.iloc[-1]))

def groupPrecipLabel(labelList):
    labels = pd.to_numeric(subsetLabels, errors='coerce', downcast='float')

    if labels.cumsum().iloc[-1] > 0 :
        return 1
    else:
        return 0

def errorCalc(predProb, groundTruth):
    mse = np.square(groundTruth - predProb).mean()
    return(mse)

if __name__ == '__main__':
    testInd = loadIndData("testIndices.txt")
    dataPath = "precip_prob_data"

    testFile = pd.read_csv(os.path.join(dataPath, "4.csv"))
    testFeatures = testFile.iloc[:,[0, 1, 2, 3, 5, 6]]
    testLabels = testFile.iloc[:,7]
    testProbs = testFile.iloc[:,4]

    n, day = 24, 1
    for i in range(0, len(testFile)-1, n):
        print("\n\nDay", day)
        subsetProbs = testProbs.iloc[i:i+n]
        maxProb = maxProbability(subsetProbs)
        jointProb = jointProbability(subsetProbs)
        subsetLabels = testLabels.iloc[i:i+n]
        groundTruth = groupPrecipLabel(subsetLabels)

        print("Predicted Probabilities")
        print("Max Prob: ", maxProb)
        print("Joint Prob: ", jointProb)
        print("Ground Truth: ", groundTruth)

        print("\nErrors")
        print("Max Prob Error:", errorCalc(maxProb, groundTruth))
        print("Joint Prob Error:", errorCalc(jointProb, groundTruth))

        day+=1
