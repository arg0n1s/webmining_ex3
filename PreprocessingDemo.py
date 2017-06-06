from bs4 import BeautifulSoup as Bs
from nltk.stem.snowball import SnowballStemmer
import codecs
import Preprocessing as Pre
from scipy import sparse

def write_to_file(path, text):
    file = codecs.open(path, 'w', 'utf-8')
    file.write(text)
    file.close()

def load_file(path):
    text = open(path, 'r', encoding='utf-8', errors='ignore').read()
    return text;

def remove_html_formating(text):
    a = Bs(text, "lxml")
    [x.extract() for x in a.findAll('script')]
    return a.get_text()

def extract_word_list(text):
    word_list = ''.join(
        [c if c.isalnum() and not c.isnumeric() else ' ' for c in text]).lower().split()
    return word_list

def load_stopwords(path):
    file = codecs.open(path, 'r', 'utf-8')
    text = file.read()
    word_list = ''.join(map(lambda c: c if c.isalnum() and not c.isnumeric() else ' ', text)).lower().split()
    return set(word_list)

def remove_stopwords(text, stopwords):
     return list(filter(lambda x: x[0] not in stopwords, text))

def stem_words(text, language):
    stemmer = SnowballStemmer(language)
    return list(map(lambda word: stemmer.stem(word), text))

def retrieve_doc_from_pickle(text, path):
    prep = Pre.load_from_disk(path)
    docs = prep.classification_objects
    for doc in docs:
        if doc.content in text:
            return doc

def extract_tf_idf_vector(doc):
    return list(map(lambda item: [item[0], item[1][2]], doc.terms.items()))

def retrieve_feature_vector(path):
    prep = Pre.load_from_disk(path)
    return prep.top_n_reoccuring_training_terms

def retrieve_sparse_feature_vector(path):
    prep = Pre.load_from_disk(path)
    mat = prep.sparse_training_data
    mat = mat.toarray()
    return sparse.csr_matrix(mat[:,0:10])

outputDir = "DemoOutput/"
path = "course-cotrain-data/fulltext/course/http_^^cs.cornell.edu^Info^Courses^Current^CS415^CS414.html"
stopwords = load_stopwords("stopwords/english")

input = load_file(path)
write_to_file(outputDir+"aufgabe2_1_1_raw.txt", input)

no_html = remove_html_formating(input)
write_to_file(outputDir+"aufgabe2_1_2_no_html.txt", no_html)

word_list = extract_word_list(no_html)
word_list_size = word_list.__len__()
write_to_file(outputDir+"aufgabe2_1_3_word_list.txt", str(word_list)+"\n Number of Words in the word list: "+str(word_list_size))

word_list_no_stopwords = remove_stopwords(word_list, stopwords)
word_list_no_stopwords_size = word_list_no_stopwords.__len__()
write_to_file(outputDir+"aufgabe2_2_1_word_list_no_stopwords.txt", str(word_list_no_stopwords)+"\n Number of Words in the word list before: "
              +str(word_list_size)+"\n After the removal of stopwords: "+str(word_list_no_stopwords_size))

word_list_stemmed = stem_words(word_list_no_stopwords, "english")
word_list_stemmed_size = word_list_stemmed.__len__()
write_to_file(outputDir+"aufgabe2_2_2_word_list_stemmed.txt", str(word_list_stemmed)+"\n Number of Words in the word list before: "
              +str(word_list_size)+"\n After the removal of stopwords: "+str(word_list_no_stopwords_size)
              +"\n After stemming: "+str(word_list_stemmed_size))

doc = retrieve_doc_from_pickle(no_html, "pre-processed-data.pickle")
tf_idf_vector = extract_tf_idf_vector(doc)
write_to_file(outputDir+"aufgabe2_3_tf_idf_vector.txt", str(tf_idf_vector)+"\n Number of Terms in tf-idf vector: "+str(tf_idf_vector.__len__()))

top_n_reoccuring_words = retrieve_feature_vector("pre-processed-data.pickle")[0:10]
write_to_file(outputDir+"aufgabe2_4_top_10_terms.txt", str(top_n_reoccuring_words))

sparse_feature_vector = retrieve_sparse_feature_vector("pre-processed-data.pickle")
write_to_file(outputDir+"aufgabe2_5_sparse_feat_vector.txt", str(sparse_feature_vector))
sparse.save_npz(outputDir+"sparse_feature_vectore_n10.npz", sparse_feature_vector)




