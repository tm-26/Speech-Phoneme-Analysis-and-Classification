import pandas
import sys

from sklearn import neighbors
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


def classifyMethod(data):
    classes = data["Class number"]
    del data["Class number"]
    del data["Gender"]
    trainData, testData, trainLabels, testLabels = train_test_split(data, classes, test_size=0.25)
    classifier = neighbors.KNeighborsClassifier(k, weights="distance")
    classifier.fit(trainData, trainLabels)
    myPrediction = classifier.predict(testData)
    return classifier, testLabels, myPrediction, testData


def classify(data):
    classifier, testLabels, myPrediction, testData = classifyMethod(data)
    print("Confusion Matrix: ")
    print(confusion_matrix(testLabels, myPrediction))
    myF1Score = f1_score(testLabels, myPrediction, average="macro")
    print("F1 score = " + str(myF1Score))
    return classifier.score(testData, testLabels), myF1Score


def classifyByGender(data, gender):
    data = data.loc[data["Gender"] == gender]
    classifier, testLabels, myPrediction, testData = classifyMethod(data)
    print("Confusion Matrix for gender " + gender)
    print(confusion_matrix(testLabels, myPrediction))
    myF1Score = f1_score(testLabels, myPrediction, average="macro")
    print("F1 score = " + str(myF1Score))
    return classifier.score(testData, testLabels), myF1Score


if __name__ == "__main__":
    # Variable Declaration
    k = 10
    numberOfLoops = 10
    groupByGender = False

    if len(sys.argv) >= 2:
        if sys.argv[1].isdigit():
            k = int(sys.argv[1])
    if len(sys.argv) >= 3:
        if sys.argv[2].isdigit():
            numberOfLoops = int(sys.argv[2])
    if len(sys.argv) >= 4:
        if sys.argv[3].replace(' ', '').lower() == "true":
            groupByGender = True

    with open("Data/ABI-1 Corpus features.csv", "r") as file:
        myData = pandas.read_csv(file,
                                 usecols=["Gender", "Class number", "Format 1 (Hz)", "Format 2 (Hz)", "Format 3 (Hz)"])

    if groupByGender:
        mPredictions = []
        fPredictions = []
        mF1Scores = []
        fF1Scores = []

        for i in range(numberOfLoops):
            print("---------Iteration " + str(i + 1) + "---------")
            prediction, F1Score = classifyByGender(myData, "F")
            fPredictions.append(prediction)
            fF1Scores.append(F1Score)
            prediction, F1Scores = classifyByGender(myData, "M")
            mPredictions.append(prediction)
            mF1Scores.append(F1Score)
            print("------------------------------\n")
        pF = sum(fPredictions) / len(fPredictions)
        pM = sum(mPredictions) / len(mPredictions)
        fF1 = sum(fF1Scores) / len(fF1Scores)
        mF1 = sum(mF1Scores) / len(mF1Scores)
        print("Female average score = " + str(pF))
        print("Male average score = " + str(pM))
        print("Classifier average score = " + str((pF + pM) / 2))
        print("Female average F1 score = " + str(fF1))
        print("Male average F1 score = " + str(mF1))
        print("F1 average score = " + str((fF1 + mF1) / 2))
    else:
        predictions = []
        F1Scores = []
        OGData = myData.copy()
        for i in range(numberOfLoops):
            print("---------Iteration " + str(i + 1) + "---------")
            prediction, F1Score = classify(myData)
            predictions.append(prediction)
            F1Scores.append(F1Score)
            myData = OGData.copy()
            print("------------------------------\n")
        print("Classifier average score = " + str(sum(predictions) / len(predictions)))
        print("F1 average score = " + str(sum(F1Scores) / len(F1Scores)))
