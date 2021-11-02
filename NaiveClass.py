import math as math
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


class NaiveBayes:

    def __init__(self):
        pass

    def separate_by_class(self,data, labels):
        separated = dict()
        for i in range(len(data)):
            vector = data[i]
            class_value = labels[i]
            if class_value not in separated:
                separated[class_value] = list()
            separated[class_value].append(vector)
        return separated

    def mean(self, num):
        return sum(num)/float(len(num))

    def std(self, num):
        avg = self.mean(num)
        variance = sum([(x-avg)**2 for x in num]) / float(len(num)-1)
        return math.sqrt(variance)

    def col_stats(self, data):
        return [(self.mean(col), self.std(col), len(col)) for col in zip(*data)]

    def stats_by_class(self, data, labels):
        separated = self.separate_by_class(data, labels)
        stats = dict()
        for class_value, row in separated.items():
            stats[class_value] = self.col_stats(row)
        return stats

    def gauss_probability(self, x, mean, std):
        e = math.exp(-((x-mean)**2 / (2 * std**2)))
        return (1 / (math.sqrt(2 * math.pi) * std)) * e

    def class_probabilities(self, stats, row):
        total_rows = sum([stats[label][0][2] for label in stats])
        probabilities = dict()
        for class_value, class_stats in stats.items():
            probabilities[class_value] = stats[class_value][0][2] / float(total_rows)
            for i in range(len(class_stats)):
                mean, std, count = class_stats[i]
                probabilities[class_value] *= self.gauss_probability(row[i], mean, std)
        return probabilities

    def predict(self, data, labels,stats):
        preds = []

        for i in range(len(data)):
            probabilities = self.class_probabilities(stats, data[i])
            preds.append(max(probabilities, key=probabilities.get))

        cf = confusion_matrix(labels, preds, labels=[5, 6])
        ax = plt.subplot()
        sns.heatmap(cf, annot=True, fmt='g', ax=ax, cbar=False,
                    cmap='Greys')

        acc_score = accuracy_score(labels, preds)
        print(f'Accuracy score is : {acc_score:.2f}')
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title(f'Confusion Matrix - Acc score : ({acc_score:.2f})')
        ax.xaxis.set_ticklabels([5, 6])
        ax.yaxis.set_ticklabels([5, 6])

        plt.show()

        acc_score = accuracy_score(labels,preds)
        print(f'Accuracy score is : {acc_score:.2f}')