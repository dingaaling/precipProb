import os
import numpy as np

testInd=[]

for line in open("testIndices.txt"):
    line = int(line.strip())
    testInd.append(line)

testInd=np.array(testInd)
print(testInd.shape, type(testInd))
