import nltk
import numpy as np
import gensim
import pickle

import re
from gensim.parsing.preprocessing import remove_stopwords
from nltk.stem import WordNetLemmatizer

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

def load_sa_model():
    with open ("../model/lr_my_word2vec.pkl", "rb") as f:
        sa_model = pickle.load(f)
    return sa_model

def load_wv_model():
    with open ("../model/my_word2vec.pkl", "rb") as f:
        wv_model = pickle.load(f)
    return wv_model

def parse_output(prediction):
    if prediction:
        return "positive"
    else:
        return "negative"
def preprocess_text(text):
    lemi = WordNetLemmatizer()
    doc = re.sub("[^a-zA-Z0-9]", " ", text)
    doc = doc.split()
    doc = [lemi.lemmatize(remove_stopwords(sentences)) for sentences in doc if remove_stopwords(sentences)]
    doc = " ".join(doc)
    sent_token = nltk.sent_tokenize(doc)
    processed_words = []
    for sent in sent_token:
        processed_words.append(simple_preprocess(sent))
    return processed_words


def generate_prediction(sa_model, wv_model ,test_sentence, score = False):
    preprocessed_text = preprocess_text(test_sentence)
    test_sentence_X = vectorize_avg_doc_my(wv_model, preprocessed_text[0])
    if score:
        return parse_output(sa_model.predict_proba([test_sentence_X])[0])
    else:
        return parse_output(sa_model.predict([test_sentence_X])[0])


def vectorize_avg_doc_my(wv_model, doc):
    return np.mean(np.array([wv_model.wv[word] for word in doc if word in wv_model.wv]), axis = 0)



if __name__ == "__main__":
    load_sa_model()
    load_wv_model()
    print("loaded!")