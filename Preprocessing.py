from pathlib import Path
import pickle
from bs4 import BeautifulSoup as bs
from nltk.stem.snowball import SnowballStemmer
from scipy import sparse
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
        self.global_terms = {}
        #self.sparse_data = sparse.csr_matrix()

    def load_files(self, directory):
        p = Path(directory)
        [self.__read_docs_from_path(x) for x in p.iterdir() if x.is_dir()]

    def safe_to_disk(self, path):
        pickle.dump(self, open(path, "wb"))

    def remove_script_tags(self):
        for doc in self.classification_objects:
            a = bs(doc.content, "lxml")
            [x.extract() for x in a.findAll('script')]
            doc.content = a.get_text()

    def extract_token_list(self):
        for doc in self.classification_objects:
            doc.token_list = ''.join([c if c.isalnum() and not c.isnumeric() else ' ' for c in doc.content]).lower().split()

    def load_stopwords(self, path):
        file = codecs.open(path, 'r', 'utf-8')
        text = file.read()
        word_list = ''.join(map(lambda c: c if c.isalnum() and not c.isnumeric() else ' ', text)).lower().split()
        self.stopwords = set(word_list)

    def remove_stopwords(self):
        #print("Removing Stopwords..")
        for doc in self.classification_objects:
            #b = doc.token_list.__len__()
            doc.token_list = list(filter(lambda x: x[0] not in self.stopwords , doc.token_list))
            #print("Before: ",b," After: ",doc.token_list.__len__())

    def stem_words(self, language):
        stemmer = SnowballStemmer(language)
        for doc in self.classification_objects:
            doc.token_list = list(map(lambda word: stemmer.stem(word), doc.token_list))

    def compute_tf(self):
        for doc in self.classification_objects:
            for word in doc.token_list:
                if word in doc.terms:
                    doc.terms[word] += 1
                else:
                    doc.terms[word] = 1

                if word in self.global_terms:
                    self.global_terms[word]= self.global_terms[word] | set([doc])
                else:
                    self.global_terms[word] = set([doc])

            for word in doc.terms:
                doc.terms[word] /= doc.token_list.__len__()

    def compute_tf_idf(self):
        for term in self.global_terms:
            docs = self.global_terms[term]
            idf = np.log(self.classification_objects.__len__() / docs.__len__())
            for doc in docs:
                doc.terms[term] = [doc.terms[term], idf, doc.terms[term] * idf]
            self.global_terms[term] = [idf, docs]

    def save_sparse_representation(self):
        self.sparse_data = sparse.csr_matrix()

    def __read_docs_from_path(self, path):
        p = Path(path)
        files = list(p.glob("*.*"))
        [self.__open_doc_and_label(x,p.name) for x in files if not ".DS" in x.name]

    def __open_doc_and_label(self, path, label):
        text = ""
        try:
            text = open(path, 'r', encoding='utf-8', errors='ignore').read()
        except Exception as inst:
            print(path)
            print(inst)
            return
        self.classification_objects.append(ClassificationObject(text, label))
        if not label in self.raw_by_labels.keys():
            self.raw_by_labels[label]=[]
        else:
            self.raw_by_labels[label].append(text)

def loadFromDisk(path):
    return pickle.load( open( path, "rb" ) )

'''
prep = Preprocessing()
prep.load_files("course-cotrain-data/fulltext")
prep.remove_script_tags()
prep.extract_token_list()
prep.load_stopwords("stopwords/english")
prep.remove_stopwords()
prep.stem_words("english")
prep.compute_tf()
prep.compute_tf_idf()
prep.safe_to_disk("pre-processed-data.pickle")
'''

prep = loadFromDisk("pre-processed-data.pickle")
print(prep.classification_objects[0].terms)