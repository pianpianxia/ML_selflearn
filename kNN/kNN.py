from os import listdir

from numpy import *


def classify_knn(newData, dataSet, dataLabel, k):
    rowNum = dataSet.shape[0]
    coordDiff = tile(newData, (rowNum, 1)) - dataSet
    diffSq = coordDiff ** 2
    distanceSq = diffSq.sum(axis=1)
    distance = distanceSq ** 0.5
    sortedDistanceIndices = distance.argsort()
    classFreq = dict()
    for i in range(k):
        classAtIndexI = dataLabel[sortedDistanceIndices[i]]
        classFreq[classAtIndexI] = classFreq.get(classAtIndexI, 0) + 1
    return max(classFreq, key=classFreq.get)


def file2matrix(fileName):
    with open(fileName) as fl:
        lines = fl.readlines()
    featureDim = len(lines[0].strip().split("\t")) - 1
    entryNum = len(lines)
    dataSet = zeros((entryNum, featureDim))
    dataLabels = []
    for i in range(entryNum):
        lineList = lines[i].strip().split("\t")
        dataSet[i, ] = lineList[0:3]
        dataLabels.append(lineList[-1])
    return dataSet, dataLabels


def rescale(dataSet):
    minVal = dataSet.min(0)
    maxVal = dataSet.max(0)
    valRange = maxVal - minVal
    dim = dataSet.shape[0]
    rescaledData = dataSet - tile(minVal, (dim, 1))
    rescaledData = rescaledData / tile(valRange, (dim, 1))
    return rescaledData, valRange, minVal


def test_knn(dataSet, dataLabel, k, hoRatio):
    rescaledData, valRange, minVal = rescale(dataSet)
    numTestData = int(hoRatio * rescaledData.shape[0])
    errorCount = 0
    predTable = zeros((numTestData, 2))
    for i in range(numTestData):
        predResult = classify_knn(rescaledData[i,], rescaledData[numTestData:, ], dataLabel[numTestData:], k)
        predTable[i, ] = predResult, dataLabel[i]
        if predResult != dataLabel[i]:
            errorCount += 1
    errorRate = errorCount / numTestData
    return errorRate, predTable


def img2vec(filename):
    with open(filename) as img:
        lines = img.readlines()
    width = len(lines)
    length = len(list(lines[0].strip()))
    vecLength = width * length
    returnVec = zeros((1, vecLength))
    for i in range(length):
        dataAtIRow = list(lines[i].strip())
        returnVec[0, i * width:(i + 1) * width] = dataAtIRow
    return returnVec


def handwriting_acquire_data(trainDirName):
    trainingLabels = []
    trainingFileList = listdir(trainDirName)
    dataNum = len(trainingFileList)
    entryLength = img2vec('%s/%s' % (trainDirName, trainingFileList[0])).shape[1]  # assuming all the files share the
    # same format
    trainingData = zeros((dataNum, entryLength))
    for i in range(dataNum):
        fileName = listdir(trainDirName)[i]
        trainingLabel = fileName[0]
        trainingLabels.append(trainingLabel)
        trainingFilePath = '%s/%s' % (trainDirName, trainingFileList[i])
        trainingData[i, ] = img2vec(trainingFilePath)
    return trainingData, trainingLabels


def handwriting_test(trainingData, trainingLabels, testDirName, k):
    errorNum = 0
    testData, testLabels = handwriting_acquire_data(testDirName)
    dataNum = testData.shape[0]
    for i in range(dataNum):
        predResult = classify_knn(testData[i, ], trainingData, trainingLabels, k)
        if predResult != testLabels[i]:
            errorNum += 1
    errorRate = errorNum / dataNum
    return errorNum, errorRate


if __name__ == "__main__":
    # group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    # labels = ['A', 'A', 'B', 'B']
    # print(classify_knn([-10, -10], group, labels, 3))
    # datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")
    # print(rescale(datingDataMat))
    # print(test_knn(datingDataMat, datingLabels, 3, 0.1))
    # a = img2vec("testDigits/0_13.txt").reshape(32, 32)
    # print(a)
    data, labels = handwriting_acquire_data("TrainingDigits")
    print(handwriting_test(data, labels, "TestDigits", 3))
