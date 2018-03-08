import os
import numpy as np
import pandas as pd

dataPath = "precip_prob_data"

# Use os.listdir (random order) to partition into training, testing, and validation
for i, csvFile in enumerate(os.listdir(dataPath)):

    if (not(csvFile.startswith("."))):

        print(i, csvFile)
        filePath = os.path.join(dataPath, str(csvFile))

        if i == 0:
            trainData = pd.read_csv(filePath)
            trainData = trainData[:-1]

        elif (i > 0 and i < 8001):
            csvData = pd.read_csv(filePath)
            csvData = csvData[:-1]
            trainData = pd.concat([trainData, csvData])

        elif i == 8001:
            testData = pd.read_csv(filePath)
            testData = testData[:-1]

        elif (i > 8001 and i < 14001):
            csvData = pd.read_csv(filePath)
            csvData = csvData[:-1]
            testData = pd.concat([testData, csvData])

        elif i == 14001:
            valData = pd.read_csv(filePath)
            valData = valData[:-1]

        else:
            csvData = pd.read_csv(filePath)
            csvData = csvData[:-1]
            valData = pd.concat([valData, csvData])

print("Train Data: ", trainData.shape)
print("Test Data: ", testData.shape)
print("Val Data: ", valData.shape)

trainData.iloc[:,1:].to_csv("trainData.csv")
testData.iloc[:,1:].to_csv("testData.csv")
valData.iloc[:,1:].to_csv("valData.csv")
