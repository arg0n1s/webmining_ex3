from pathlib import Path
import pickle
from bs4 import BeautifulSoup as bs

class ClassificationObject:
    def __init__(self, content, label):
        self.content = content
        self.label = label
        self.token_list = []

class Preprocessing:
    def __init__(self):
        self.classification_objects = []
        self.raw_by_labels = {}

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

prep = Preprocessing()
prep.load_files("course-cotrain-data/fulltext")
prep.remove_script_tags()
prep.extract_token_list()
prep.safe_to_disk("pre-processed-data.pickle")

#prep = loadFromDisk("pre-processed-data.pickle")