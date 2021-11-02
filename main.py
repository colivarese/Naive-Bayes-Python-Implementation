import numpy as np
import pandas as pd
from utils import *
from NaiveClass import NaiveBayes
import matplotlib.pyplot as plt

if __name__ == '__main__':
    
    trainData, trainLabels, testData, testLabels = load_example_data()

    NaiveBayes = NaiveBayes()

    class_stats = NaiveBayes.stats_by_class(trainData,trainLabels)

    NaiveBayes.predict_example(testData, testLabels,class_stats)

