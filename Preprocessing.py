from pathlib import Path
import pickle
from bs4 import BeautifulSoup as Bs
from nltk.stem.snowball import SnowballStemmer
from scipy import sparse
from random import shuffle
import numpy as np
import codecs


class ClassificationObject:
    def __init__(self, content, label):
        self.content = content
        self.label = label
        self.token_list = []
        self.terms = {}


class Preprocessing:
    def __init__(self):
        self.classification_objects = []
        self.raw_by_labels = {}
        self.stopwords = set()
        self.global_training_terms = {}
        self.top_n_reoccuring_training_terms = ()
        self.training_set = []
        self.test_set = []
        self.sparse_training_data = []
        self.sparse_testing_data = []
        self.training_labels = []
        self.testing_labels = []

    def load_files(self, directory):
        p = Path(directory)
        [self.__read_docs_from_path(x) for x in p.iterdir() if x.is_dir()]

    def safe_to_disk(self, path):
        pickle.dump(self, open(path, "wb"))

    def remove_script_tags(self):
        for doc in self.classification_objects:
            a = Bs(doc.content, "lxml")
            [x.extract() for x in a.findAll('script')]
            doc.content = a.get_text()

    def extract_token_list(self):
        for doc in self.classification_objects:
            doc.token_list = ''.join(
                [c if c.isalnum() and not c.isnumeric() else ' ' for c in doc.content]).lower().split()

    def load_stopwords(self, path):
        file = codecs.open(path, 'r', 'utf-8')
        text = file.read()
        word_list = ''.join(map(lambda c: c if c.isalnum() and not c.isnumeric() else ' ', text)).lower().split()
        self.stopwords = set(word_list)

    def remove_stopwords(self):
        # print("Removing Stopwords..")
        for doc in self.classification_objects:
            # b = doc.token_list.__len__()
            doc.token_list = list(filter(lambda x: x[0] not in self.stopwords, doc.token_list))
            # print("Before: ",b," After: ",doc.token_list.__len__())

    def stem_words(self, language):
        stemmer = SnowballStemmer(language)
        for doc in self.classification_objects:
            doc.token_list = list(map(lambda word: stemmer.stem(word), doc.token_list))

    def build_training_and_test_sets(self):
        if self.classification_objects.__len__() < 230:
            print("Error: The data set has to have more than 230 elements")
        else:
            indices = [i for i in range(self.classification_objects.__len__())]
            shuffle(indices)
            if indices.__len__() < 320:
                training_indices = indices[0:160]
                test_indices = indices[160:]
            else:
                training_indices = indices[0:int(np.ceil(indices.__len__() / 2))]
                test_indices = indices[int(np.ceil(indices.__len__() / 2)):]

            self.training_set = [self.classification_objects[i] for i in training_indices]
            self.test_set = [self.classification_objects[i] for i in test_indices]

    def compute_tf(self, training):
        if training is True:
            dataset = self.training_set
        else:
            dataset = self.test_set

        for doc in dataset:
            for word in doc.token_list:
                if word in doc.terms:
                    doc.terms[word] += 1
                else:
                    doc.terms[word] = 1

                if training is True and word in self.global_training_terms:
                    self.global_training_terms[word] = self.global_training_terms[word] | {doc}
                elif training is True:
                    self.global_training_terms[word] = {doc}

            for word in doc.terms:
                doc.terms[word] /= doc.token_list.__len__()

    def compute_tf_idf(self):

        for term in self.global_training_terms:
            docs = self.global_training_terms[term]
            idf = np.log(self.training_set.__len__() / docs.__len__())
            for doc in docs:
                doc.terms[term] = [doc.terms[term], idf, doc.terms[term] * idf]
            self.global_training_terms[term] = [idf, docs.__len__(), docs]

            #print("TERM: ",term," // IDF: ",self.global_training_terms[term][0]," // RANK: ",self.global_training_terms[term][1])

    def compute_top_n_reoccuring_terms(self, n):
        sorted_words = sorted(self.global_training_terms.items(), key=lambda item: item[1][1], reverse=True)
        selected_features = list(map(lambda item: [item[0], item[1][1]], sorted_words))
        self.top_n_reoccuring_training_terms = selected_features[0:n]

    def make_sparse(self, training):
        dataset = self.training_set if training is True else self.test_set
        sparse_data = []
        labels = []
        for doc in dataset:
            labels.append(doc.label)
            feature_vector = []
            for feature in self.top_n_reoccuring_training_terms:
                if not feature[0] in doc.terms:
                    feature_vector.append(0.0)
                elif training is True:
                    feature_vector.append(doc.terms[feature[0]][2])
                else:
                    feature_vector.append(doc.terms[feature[0]] * self.global_training_terms[feature[0]][0])

            sparse_data.extend([feature_vector])

        if training is True:
            self.sparse_training_data = sparse.csr_matrix(sparse_data)
            self.training_labels = labels
        else:
            self.sparse_testing_data = sparse.csr_matrix(sparse_data)
            self.testing_labels = labels

    def __read_docs_from_path(self, path):
        p = Path(path)
        files = list(p.glob("*.*"))
        [self.__open_doc_and_label(x, p.name) for x in files if ".DS" not in x.name]

    def __open_doc_and_label(self, path, label):
        try:
            text = open(path, 'r', encoding='utf-8', errors='ignore').read()
        except Exception as inst:
            print(path)
            print(inst)
            return
        self.classification_objects.append(ClassificationObject(text, label))
        if label not in self.raw_by_labels.keys():
            self.raw_by_labels[label] = []
        else:
            self.raw_by_labels[label].append(text)


def load_from_disk(path):
    return pickle.load(open(path, "rb"))
