import pandas as pd
import numpy as np

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