import Preprocessing as Pre
import TrainingAndEvaluation as Tr
import time
import matplotlib.pyplot as plt

'''
prep = Pre.Preprocessing()
prep.load_files("course-cotrain-data/fulltext")
prep.remove_script_tags()
prep.extract_token_list()
prep.load_stopwords("stopwords/english")
prep.remove_stopwords()
prep.stem_words("english")
prep.build_training_and_test_sets()
# Training
prep.compute_tf(True)
prep.compute_tf_idf()
prep.compute_top_n_reoccuring_terms(100)
prep.make_sparse(True)
# Testing
prep.compute_tf(False)
prep.make_sparse(False)
prep.safe_to_disk("pre-processed-data.pickle")
'''
prep = Pre.load_from_disk("pre-processed-data.pickle")
training_and_eval = Tr.TrainingAndEvaluation(prep)
t0 = time.time()
training_and_eval.training()
t1 = time.time()
print("*** Results when taking 100 features ***\n")
print("Elapsed time during training: " + str(t1 - t0) + " seconds")
t0 = time.time()
training_and_eval.testing(False)
t1 = time.time()
print("Elapsed time during testing: " + str(t1 - t0) + " seconds")
training_and_eval.evaluate()
print("True positives: " + str(training_and_eval.tp))
print("False positives: " + str(training_and_eval.fp))
print("False negatives: " + str(training_and_eval.fn))
print("True negatives: " + str(training_and_eval.tn) + '\n')
print("Accuracy with the SVM classifier: " + str(training_and_eval.accuracy) + '\n')
training_and_eval.testing(True)
training_and_eval.evaluate()
print("Accuracy with the majority-class classifier: " + str(training_and_eval.accuracy) + '\n')
training_and_eval.compute_probabilities()
training_and_eval.compute_precision_recall_curve()
training_and_eval.sweep_number_of_features()
training_and_eval.plot_performance_analysis()
plt.show()