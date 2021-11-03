from typing import Any
import pandas as pd
import numpy as np

def separate_data(data:pd.DataFrame, outcome:str, percent:float):
    size = len(data)
    div = size * percent
    low = int(np.floor(div))
    lim = size - low
    training_data = data.iloc[:lim]
    test_data = data.iloc[lim:]

    training_labels = training_data[outcome]
    test_labels = test_data[outcome]

    training_data = training_data.drop(columns=[outcome])
    test_data = test_data.drop(columns=[outcome])


    training_data = training_data.to_numpy()
    test_data = test_data.to_numpy()
    training_labels = training_labels.to_numpy()
    test_labels = test_labels.to_numpy()

    return training_data, test_data, training_labels, test_labels


def load_example_data():

    path = './dataset/'

    for _ in range(1):
        # Load training data
        trainData = pd.read_csv(path + 'trainData1.csv').to_numpy()
        tmp = pd.read_csv(path + 'trainData1.csv').columns
        tmp = [float(i) for i in tmp]
        trainData = np.insert(trainData, 0, tmp, axis=0)

        trainLabels = pd.read_csv(path + 'trainLabels1.csv').to_numpy()
        tmp = pd.read_csv(path + 'trainLabels1.csv').columns
        tmp = [float(i) for i in tmp]
        trainLabels = np.insert(trainLabels, 0, tmp, axis=0)
        trainLabels = trainLabels[:, 0]

        for i in range(2,11):
            tmp_data = pd.read_csv(path + f'trainData{i}.csv').to_numpy()
            tmp = pd.read_csv(path + f'trainData{i}.csv').columns
            tmp = [float(i) for i in tmp]
            tmp_data = np.insert(tmp_data, 0, tmp, axis=0)
            trainData = np.concatenate((trainData,tmp_data))

            tmp_labels = pd.read_csv(path + f'trainLabels{i}.csv').to_numpy()
            tmp = pd.read_csv(path + f'trainLabels{i}.csv').columns
            tmp = [float(i) for i in tmp]
            tmp_labels = np.insert(tmp_labels, 0, tmp, axis=0)
            tmp_labels = tmp_labels[:, 0]
            trainLabels = np.concatenate((trainLabels, tmp_labels))

        testData = pd.read_csv(path + 'testData.csv').to_numpy()
        tmp = pd.read_csv(path + 'testData.csv').columns
        tmp = [float(i) for i in tmp]
        testData = np.insert(testData, 0, tmp, axis=0)

        testLabels = pd.read_csv(path + 'testLabels.csv').to_numpy()
        tmp = pd.read_csv(path + 'testLabels.csv').columns
        tmp = [float(i) for i in tmp]
        testLabels = np.insert(testLabels, 0, tmp, axis=0)

    return trainData, trainLabels, testData, testLabels