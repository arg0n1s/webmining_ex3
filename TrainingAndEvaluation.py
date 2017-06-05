from sklearn import svm


class TrainingAndEvaluation:
    def __init__(self, preprocessed_data):
        self.prep_data = preprocessed_data
        self.classifier = svm.SVC(C=50.0, kernel='linear', probability=True)
        self.preditions = []
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.accuracy = 0
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
