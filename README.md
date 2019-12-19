# precipProb

This project uses the DarkSky dataset of 16,144 7-day hourly forecast weather data to generate models that predict the precipitation probability for a given "n" number of hours. <br>

For testing purposes, the dataset was randomly split into 8,000 training, 6,000 testing, and 2,144 validation csv sample sets. From these sets, hourly forecast data is aggregated into `trainData.csv`, `valData.csv`, and `testData.csv`. <br>

The "final" folder contains the methods necessary to implement precipitation prediction. The user sets parameters in `inputFile.txt` for the name of the csv file with the testing data desired, the "n" number of hours included in the precipitation prediction, and the kind of feature descriptor for the data. Options include all of the forecast features, a precipitation probability count distribution, and simply the list of precipitation probabilities. <br>

The "n" number of hours specifies the time range for the precipitation probability prediction. The "n" should be an integer in the range of 1 to 168, inclusive. An "n" larger than 168 shouldn't be chosen as that would cause data from different weather stations to overlap. Based on the "n" provided, the features will be re-grouped to fit that size. The labels (actual precipitation results, or didPrecip in the dataset) will also be grouped and if there was precipitation in that time range, the label for the group will be 1. Otherwise it will be 0. <br>

Finally, based on the feature type, different models will be applied to calculate the precipitation probability. For all forecast and probability distribution feature types, three supervised machine learning models are implemented: Logistic Regression, Multi-Layer Perceptron, and Random Forest. These are implemented using the Sci-kit Learn library and return prediction and display score and mean squared error. For the probability list feature, basic statistical methods are used: maximum, joint, and average. These display mean squared error. For all feature/model types, the three prediction results from the three different models are averaged to generate a final precipitation prediction probability which is saved to the variable precipProbPred and printed to the console. <br>

`TestResults.jpg` has quantified results, but some interesting notes: <br>
1. n = 1 produced the highest levels of accuracy, and by n=24, accuracy had dropped about 10% for ML models
2. Baseline statistical models tend to overemphasize probability (although joint probability diminishes rapidly as "n" increases). However, their Mean Squared Error (MSE) was still comparable to the ML-trained models <br>
3. Random Forest tended to outperform in experimentation, but MLP performed slightly better in testing. <br>
4. Averaging the results of all three models uniformly reduced MSE and served as a weak form of boosting <br>

#### Output Format

```
[Model Type] [Feature Type] [Time Interval (hours)]
Model Score: [Correct (out of 1)] MSE: [Mean Squared Error]
```

Default parameters (Supervised Model + allFeatures) were chosen because they produced the lowest MSE.

Future investigations: <br>
1. Weighted boosting for different features and models, depending on their accuracy in different situations. <br>
2. Increase dataset and implement LSTM RNN, which may increase accuracy for larger "n" tests <br>

## Final Method - Documentation

### precipProp.py
Main method to predict probability of precipitation in a given time range <br>
Input - n x 7 feature vector for n # of hours <br>
Output - final precipitation probability prediction & score + mean squared error results <br>
Example command: python3 precipProp.py inputFile.txt <br>
Input file can have any name, as long it is called correctly. <br>
If you would like the model scores and errors to not be printed, find the printScore parameter in runModel and change to printScore=False <br>

### inputFile.txt
Parameters for the prediction model <br>
1. Name of test file (ex. testData.csv) <br>
2. Size of prediction grouping (ex. 48, meaning precipitation probability for a 48-hour range. Must be integer 1 or larger) <br>
3. Feature type: <br>
   -allFeatures: all of the 7 forecast features for each hour are used <br>
   -probDistrFeatures: a probability distribution bag of words vector is generated for the precipitation probabilities <br>
   -probList: list of precipitation probabilities <br>
Examples for `inputFile.txt`: <br>

*Default Parameters* <br>
testData.csv <br>
24 <br>
allFeatures <br>

*Example 1 Parameters* <br>
sampleData.csv <br>
48 <br>
probDistrFeatures <br>

These are the parameters chosen for demo because they show clearly the prediction probability for a 48-hour time range. <br>

*Example 2 Parameters* <br>
valData.csv <br>
2 <br>
probList <br>

### dataTransforms.py
Functions to clean, pre-process, and group feature and label data

### model.py
Functions to apply different models to make precipitation probability <br>
1. Supervised: Logistic Regression, MLP, and Random Forest <br>
   Used for allFeatures and probDistrFeatures type <br>
2. Baseline: max, joint, and average probabilties for a different group <br>
   Used for raw probability list feature type <br>

### partitionData.py
Splits the 16144-length dataset randomly into training (8000), testing (6000), and validation (2214) <br>
Can be used to generate a new random set of trainData, testData, and valData

## Exploratory - Documentation

### TestResults.jpg
Accuracy and mean-squared error scores of each model for n = 1, 12, 24, and 48 :<br>
1. Supervised (LR, MLP, RF) with all 7 weather features <br>
2. Supervised (LR, MLP, RF) with probability distribution Bag of Words-like vector <br>
3. Baseline statistical methods on probability grouping (max, average, joint) <br>
The MSE of the Final prediction (average of all three model predictions) is also shown <br>

### randomInd.py
Exploratory method of setting random indices

### baseline.py
Baseline probability calculation technique takes hourly precipProbability and "n" number of hours as input and generates a probability score. Classic techniques of 1) taking max(set of hourly probabilities) and 2) multiplying precipProbabilities to generate a joint probability are used. These are simplistic and make false assumptions (ex. event independence), but serve as a starting point to compare future methods. 

This file also contains error calculation methods (mean squared error and sum of squared difference) to allow quantifiable comparison

### supervised.ipynb
Groups data based on time range (ex. 24 for 24 hours in a day) and applies sklearn models -  Logistic Regression and MLP

### rnn.ipynb
Exploratory method to implement an LSTM RNN
