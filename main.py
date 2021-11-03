import numpy as np
import pandas as pd
from utils import *
from NaiveClass import NaiveBayes
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

if __name__ == '__main__':
    
    #trainData, trainLabels, testData, testLabels = load_example_data()

    #NaiveBayes = NaiveBayes()

    #class_stats = NaiveBayes.stats_by_class(trainData,trainLabels)

    #NaiveBayes.predict_example(testData, testLabels,class_stats)

    data = pd.read_csv('./dataset/GAD_NB.csv')
    training_data, test_data, training_labels, test_labels = separate_data(data=data, outcome='GAD_cat',percent=0.3)

    NaiveBayes = NaiveBayes()

    class_stats = NaiveBayes.stats_by_class(training_data, training_labels)
    NaiveBayes.predict(test_data, test_labels, class_stats)
   
