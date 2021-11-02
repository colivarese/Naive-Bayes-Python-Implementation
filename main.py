import numpy as np
import pandas as pd
from utils import *
from NaiveClass import NaiveBayes
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    trainData, trainLabels, testData, testLabels = load_example_data()

    NaiveBayes = NaiveBayes()

    class_stats = NaiveBayes.stats_by_class(trainData,trainLabels)

    NaiveBayes.predict(testData, testLabels,class_stats)

    for n in [5,6]:
        img_mean= []
        for c in class_stats[n]:
            img_mean.append(c[0])
        img_mean = np.array([int(x) for x in img_mean])
        img_five = img_mean.reshape((8,8))
        plt.figure(figsize=(4,4))
        plt.imshow(img_five)
        plt.show()
    