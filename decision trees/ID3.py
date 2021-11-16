from math import log


def calEntropy(dataSet):
    """
    Return the Shannon entropy of the data set.

    Note : This function is assuming the given dataSet has its labels located at the last row.

    Parameters
    ----------
    dataSet : list
        The list that contains the data which needs to calculate the entropy.

    Returns
    -------
    entropy : float
        The float number that represents the entropy.
    """
    entryNum = len(dataSet)
    labelFreq = dict()
    for entry in dataSet:
        curLabel = entry[-1]
        labelFreq[curLabel] = labelFreq.get(curLabel, 0) + 1
    entropy = 0.0
    for key in labelFreq:
        prob = labelFreq[key] / entryNum
        entropy -= prob * log(prob, 2)
    return entropy


def splitDataSet(dataSet, col, value):
    """
    Return the sub-data set split by the given column number and the given value of the column.

    Note : This function is assuming the given dataSet has its labels located at the last row.

    Parameters
    ----------
    dataSet : list
        The list that contains the data which needs to calculate the entropy.
    col : int
        The integer that represents the column number that need to be split.
    value : int or str
        The value in the selected column that is chosen to remain in the resultDataset.

    Returns
    -------
    resultDataset : list
        The list that contains the split data.
    """
    resultDataset = []
    for entry in dataSet:
        if entry[col] == value:
            resultDataset.append(entry[:col] + entry[col + 1:])
    return resultDataset


def findBestColToSplit(dataSet):
    """
    Return the column number of the data set that minimize the entropy after split.

    Note : This function is assuming the given dataSet has its labels located at the last row.

    Parameters
    ----------
    dataSet : list
        The list that contains the data which needs to find the best possible column to split.

    Returns
    -------
    bestCol : int
        The integer that represents the column number that optimize the entropy reduction process.
    """
    featureNum = len(dataSet[0]) - 1  # the value is subtracted by one since the last column is for labels
    entryNum = len(dataSet)
    origEntropy = calEntropy(dataSet)
    dataTranspose = list(zip(*dataSet))  # acquire the data in each column
    bestDiff = 0.0
    bestCol = -1
    for i in range(featureNum):
        caseInColI = set(dataTranspose[i])  # acquire all possible outcomes in the selected column
        newEntropy = 0.0
        for case in caseInColI:
            subDataSet = splitDataSet(dataSet, i, case)
            subEntropy = calEntropy(subDataSet)
            subProb = len(subDataSet) / entryNum
            newEntropy += subProb * subEntropy
        entropyDiff = origEntropy - newEntropy
        if entropyDiff >= bestDiff:
            bestDiff = entropyDiff
            bestCol = i
    return bestCol


def majorityVote(classVotes):
    """
    Return the class that has been voted by the most.

    Parameters
    ----------
    classVotes : list or tuple
        The lsit that records the number of votes in each class.

    Returns
    -------
    class : str / int
        The integer or string that represents the class that has been voted by the most.

    Examples
    -------
    # >>> majorityVote(['yes', 'yes', 'yes', 'no', 'no', 'maybe'])
    'yes'
    """
    classDict = dict()
    for item in classVotes:
        classDict[item] = classDict.get(item, 0) + 1
    return max(classDict, key=classDict.get)


def createTree(dataSet, featureName):
    """
    Return decision tree created using ID3 algorithm.

    Note : This function is assuming the given dataSet has its labels located at the last row.

    Parameters
    ----------
    dataSet : list
        The list that contains the data which needs to find the best possible column to split.
    featureName : list
        The list that contains the name of each column in the data set so that the tree is more readable.

    Returns
    -------
    tree : dict
        The dictionary that represents the decision tree.
    """
    dataTranspose = list(zip(*dataSet))
    if len(dataSet[0]) == 2:  # if there is only one attribute in the data set
        cases = set(dataTranspose[0])  # acquire the cases of the only attribute column
        result = {featureName[0]: dict()}
        for case in cases:
            subDataSet = splitDataSet(dataSet, 0, case)
            labels = list(zip(*subDataSet))[-1]  # acquire the label column
            if labels.count(labels[0]) == len(labels):  # only one type of label is shown in this case
                result[featureName[0]][case] = labels[0]
            else:  # if got more than one type of label, use the one that has the largest number
                result[featureName[0]][case] = majorityVote(labels)
    else:
        splitCol = findBestColToSplit(dataSet)
        splitFeatureName = featureName[splitCol]  # find the best column to split and its name
        result = {splitFeatureName: dict()}
        cases = set(dataTranspose[splitCol])  # acquire the cases of the selected attribute column
        subFeatureName = featureName[:splitCol] + featureName[splitCol + 1:]  # update the column name table
        for case in cases:
            subDataSet = splitDataSet(dataSet, splitCol, case)
            subLabels = list(zip(*subDataSet))[-1]  # acquire the label column
            if subLabels.count(subLabels[0]) == len(subLabels):
                result[splitFeatureName][case] = subLabels[0]
            else:
                result[splitFeatureName][case] = createTree(subDataSet, subFeatureName)
    return result


if __name__ == "__main__":
    fish = [[1, 1, 0, 'yes'],
            [1, 1, 1, 'no'],
            [1, 0, 0, 'yes'],
            [0, 1, 1, 'no'],
            [0, 1, 0, 'no'],
            [0, 0, 0, 'yes']]
    temp = [[1, 'yes'],
            [1, 'yes'],
            [0, 'no'],
            [0, 'yes'],
            [0, 'no']]
    name = ['no surfacing', 'flippers', 'lol']
    # print(calEntropy(fish))
    # print(splitDataSet(fish, 0, 0))
    # print(findBestColToSplit(fish))
    # print(majorityVote(['yes', 'yes', 'yes', 'no', 'no', 'maybe']))
    print(createTree(fish, name))
