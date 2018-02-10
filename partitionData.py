import os, sys
import numpy as np

# Partition training (8000), validation (2144), and testing (6000) dataset
def partition(count):
    indices = np.arange(count)
    np.random.shuffle(indices)
    trainInd, valInd, testInd = indices[:8000], indices[8000:14000], indices[14000:]
    return indices, trainInd, valInd, testInd

# Count number of files
def fileCount(dataPath):
    count=0
    for csv in os.listdir(dataPath):
        if(not(csv.startswith("."))):
            count+=1
    return count

def createTxt(filename, indices):

    file = open(filename, "w", errors="ignore")

    for ind in indices:
        ind = str(ind)
        file.write(ind + "\n")

if __name__ == '__main__':
    dataPath = "precip_prob_data"
    numFiles = fileCount(dataPath)
    indices, trainInd, valInd, testInd = partition(numFiles)
    print(indices.shape, trainInd.shape, valInd.shape, testInd.shape)

    createTxt("indices.txt", indices)
    createTxt("trainIndices.txt", trainInd)
    createTxt("valIndices.txt", valInd)
    createTxt("testIndices.txt", testInd)
