from sklearn import svm

class TrainingAndEvaluation:
    def __init__(self, preprocessed_data):
        self.prep_data = preprocessed_data
        self.classifier = svm.SVC(C=50.0, kernel='linear')
        self.preditions = []
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.accuracy = 0

    def training(self):
        self.classifier.fit(self.prep_data.sparse_training_data, self.prep_data.training_labels)

    def testing(self):
        self.predictions = self.classifier.predict(self.prep_data.sparse_testing_data)

    def evaluate(self):
        for i in range(0, self.predictions.__len__()):
            if self.predictions[i] == 'course' and self.prep_data.testing_labels[i] == 'course':
                self.tp += 1
            elif self.predictions[i] == 'non-course' and self.prep_data.testing_labels[i] == 'course':
                self.fn += 1
            elif self.predictions[i] == 'course' and self.prep_data.testing_labels[i] == 'non-course':
                self.fp += 1
            elif self.predictions[i] == 'non-course' and self.prep_data.testing_labels[i] == 'non-course':
                self.tn += 1

        self.accuracy = (self.tp + self.tn)/(self.tp + self.tn + self.fn + self.fp)

        print("True positives: " + str(self.tp))
        print("False positives: " + str(self.fp))
        print("False negatives: " + str(self.fn))
        print("True negatives: " + str(self.tn))
        print("Accuracy: " + str(self.accuracy))