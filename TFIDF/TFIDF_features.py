import pickle
import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer


def TFIDF_features(data, mode):
    list_sentence1 = data[0][0]
    list_sentence2 = data[1][0]
    list_gold_label = data[2][0]

    train_corpus_sentence1 = [' '.join(item) for item in list_sentence1]
    train_corpus_sentence2 = [' '.join(item) for item in list_sentence2]
    num_samples = len(train_corpus_sentence1)

    train_corpus = [train_corpus_sentence1[ind] + " " + train_corpus_sentence2[ind] for ind in range(num_samples)]

    if mode == "train":
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

    """
    tfidf_feature_array = scipy.sparse.csc_matrix.multiply(tfidf_sentecnce1, tfidf_sentecnce2)
    """

    """
    tfidf_distance = tfidf_sentecnce1 - tfidf_sentecnce2
    tfidf_feature = [np.linalg.norm(tfidf_distance[ind].toarray()) for ind in range(tfidf_distance.shape[0])]
    tfidf_feature_array = np.asarray(tfidf_feature).reshape(-1, 1)
    """

    tfidf_feature_array = TFIDF_vect.transform(train_corpus)

    tfidf_label = [0 if item == "contradiction" else 1 if item == "neutral" else 2 if item == "entailment" \
                    else -1 for item in list_gold_label]

    return tfidf_feature_array, tfidf_label
