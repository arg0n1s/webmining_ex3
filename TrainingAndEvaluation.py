from sklearn import svm
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np
import time


class TrainingAndEvaluation:
    def __init__(self, preprocessed_data):
        self.prep_data = preprocessed_data
        self.classifier = svm.SVC(C=50.0, kernel='linear', probability=True)
        self.predictions = []
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.accuracy = 0
        self.probabilities = []
        self.accuracy_sweep_results = {}
        self.n_features = []
        # Compute the class frequencies for the majority class classifier
        classes = {}
        for label in self.prep_data.training_labels:
            if label in classes:
                classes[label] += 1
            else:
                classes[label] = 1

        classes = sorted(classes.items(), key=lambda c: c[1], reverse=True)
        self.majority_class = classes[0][0]
        self.secondary_class = classes[1][0]

    def training(self):
        self.classifier.fit(self.prep_data.sparse_training_data, self.prep_data.training_labels)

    def testing(self, use_majority_class_classifier=False):
        self.predictions = [self.majority_class for i in range(0, self.prep_data.sparse_testing_data.shape[0])] \
            if use_majority_class_classifier is True else self.classifier.predict(self.prep_data.sparse_testing_data)

    def evaluate(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0
        self.accuracy = 0.0

        for i in range(self.predictions.__len__()):
            if self.predictions[i] == self.secondary_class and self.prep_data.testing_labels[i] == self.secondary_class:
                self.tp += 1
            elif self.predictions[i] == self.majority_class and self.prep_data.testing_labels[i] == self.secondary_class:
                self.fn += 1
            elif self.predictions[i] == self.secondary_class and self.prep_data.testing_labels[i] == self.majority_class:
                self.fp += 1
            elif self.predictions[i] == self.majority_class and self.prep_data.testing_labels[i] == self.majority_class:
                self.tn += 1

        self.accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fn + self.fp)

    def compute_probabilities(self):
        self.probabilities = self.classifier.predict_proba(self.prep_data.sparse_testing_data)

    def compute_precision_recall_curve(self):
        binary_labels = label_binarize(self.prep_data.testing_labels, classes=['course', 'non-course'])
        precision = dict()
        recall = dict()
        for i in range(2):
            precision[i], recall[i], _ = precision_recall_curve(binary_labels, self.probabilities[:, i])

        # Plot Precision-Recall curve for each class
        plt.clf()
        for i in range(2):
            classname = 'course' if i is 0 else 'non-course'
            plt.plot(recall[i], precision[i], label='Precision-recall curve of class ' + classname)
            x_values = np.array(recall[i])
            y_values = np.array(precision[i])
            idx = np.argmin(np.abs(x_values - y_values)[1:])
            plt.scatter(x_values[idx], y_values[idx], marker='o', label='Break-even point of class ' + classname + ' at {:.{prec}f}'.format(precision[i][idx], prec=3))

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve')
        plt.legend(loc="lower right")

    def sweep_number_of_features(self):
        self.n_features = [5]
        i = 5
        while i * 2 <= self.prep_data.global_training_terms.__len__():
            i *= 2
            self.n_features.append(i)

        for n in self.n_features:
            self.prep_data.compute_top_n_reoccuring_terms(n)
            self.prep_data.make_sparse(True)
            self.prep_data.compute_tf(False)
            self.prep_data.make_sparse(False)
            t0 = time.time()
            self.training()
            t1 = time.time()
            self.testing(False)
            self.evaluate()
            self.accuracy_sweep_results[n] = [self.accuracy, t1 - t0]

    def plot_performance_analysis(self):
        accuracy_values = list(map(lambda item: item[1][0], self.accuracy_sweep_results.items()))
        elapsed_times = list(map(lambda item: item[1][1] * 1000.0, self.accuracy_sweep_results.items()))
        plt.figure()
        plt.plot(self.n_features, accuracy_values)
        plt.scatter(self.n_features, accuracy_values, edgecolors='red')
        plt.xlim([4, self.n_features[-1]])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Number of features')
        plt.ylabel('Accuracy')
        plt.title('Number of features vs. Accuracy')
        plt.legend(loc="lower right")

        plt.figure()
        plt.plot(self.n_features, elapsed_times)
        plt.scatter(self.n_features, elapsed_times, edgecolors='red')
        plt.xlim([4, self.n_features[-1]])
        plt.ylim([0.0, 1000.0])
        plt.xlabel('Number of features')
        plt.ylabel('Elapsed time in ms')
        plt.title('Number of features vs. Elapsed time')
        plt.legend(loc="lower right")