import Preprocessing as pre
import TrainingAndEvaluation as tr
import time

'''
prep = pre.Preprocessing()
prep.load_files("course-cotrain-data/fulltext")
prep.remove_script_tags()
prep.extract_token_list()
prep.load_stopwords("stopwords/english")
prep.remove_stopwords()
prep.stem_words("english")
prep.build_training_and_test_sets()
# Training
prep.compute_tf(True)
prep.compute_tf_idf(True)
prep.compute_top_n_reoccuring_terms(1000, True)
prep.make_sparse(True)
# Testing
prep.compute_tf(False)
prep.compute_tf_idf(False)
prep.compute_top_n_reoccuring_terms(1000, False)
prep.make_sparse(False)
prep.safe_to_disk("pre-processed-data.pickle")
'''
prep = pre.loadFromDisk("pre-processed-data.pickle")
training_and_eval = tr.TrainingAndEvaluation(prep)
t0 = time.time()
training_and_eval.training()
t1 = time.time()
print("Elapsed time during training: " + str(t1 - t0) + " seconds")
t0 = time.time()
training_and_eval.testing()
t1 = time.time()
print("Elapsed time during testing: " + str(t1 - t0) + " seconds")
training_and_eval.evaluate()
#print(prep.testing_labels)
#print(prep.sparse_training_data.toarray())