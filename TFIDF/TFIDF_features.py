import pickle
import numpy
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer


def TFIDF_features(data, mode):
    list_sentence1 = data[0][0]
    list_sentence2 = data[1][0]
    list_gold_label = data[2][0]

    train_corpus_sentence1 = [' '.join(item) for item in list_sentence1]
    train_corpus_sentence2 = [' '.join(item) for item in list_sentence2]

    if mode == "train":
        train_corpus = train_corpus_sentence1 + train_corpus_sentence2

        TFIDF_vect = TfidfVectorizer()
        TFIDF_vect.fit(train_corpus)

        with open('./model/TFIDF.pickle', "wb") as file:
            pickle.dump(TFIDF_vect, file)

    elif mode == "test":
        with open('./model/TFIDF.pickle', "rb") as file:
            TFIDF_vect = pickle.load(file)

    else:
        print("Invalid mode selection")
        exit(0)

    tfidf_sentecnce1 = TFIDF_vect.transform(train_corpus_sentence1)
    tfidf_sentecnce2 = TFIDF_vect.transform(train_corpus_sentence2)

    tfidf_feature = scipy.sparse.csc_matrix.multiply(tfidf_sentecnce1, tfidf_sentecnce2)

    tfidf_label = [0 if item == "contradiction" else 1 if item == "neutral" else 2 if item == "entailment" \
        else -1 for item in list_gold_label]

    return tfidf_feature, tfidf_label
