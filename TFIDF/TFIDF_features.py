import pickle
import numpy as np
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer


# Process the data and return the TFIDF features and the labels for the data
def TFIDF_features(data, mode):
    # Get the sentences and labels from composite data
    list_sentence1 = data[0][0]
    list_sentence2 = data[1][0]
    list_gold_label = data[2][0]

    # Merge each sublist (tokens list of each sentence) to a string
    corpus_sentence1 = [' '.join(item) for item in list_sentence1]
    corpus_sentence2 = [' '.join(item) for item in list_sentence2]
    num_samples = len(list_gold_label)

    # Create a composite corpus over which to train the TFIDF Vectorizer
    # Corresponding lines of sentence1 and sentence2 are merged together
    corpus = [corpus_sentence1[ind] + " " + corpus_sentence2[ind] for ind in range(num_samples)]

    # There are entries in dataset without any gold_label (with gold_label entry as "-")
    # I choose to delete those from my data as they do not provide any information
    del_list =[]
    tfidf_labels = [None] * num_samples
    for ind, item in enumerate(list_gold_label):
        if item == "contradiction":
            tfidf_labels[ind] = 0
        elif item == "neutral":
            tfidf_labels[ind] = 1
        elif item == "entailment":
            tfidf_labels[ind] = 2
        else:
            tfidf_labels[ind] = 99
            del_list.append(ind)

    # Delete entries with gold_label "-"
    del_list.sort(reverse=True)
    for ind in del_list:
        del corpus[ind]
        del corpus_sentence1[ind]
        del corpus_sentence2[ind]
        del tfidf_labels[ind]

    # If mode is training we fit our TFIDF Vectorizer over our composite corpus and store it in
    # pickle format. During testing time, we retrieve this same vectorizer to generate TFIDF
    # representations for out text input
    if mode == "train":
        TFIDF_vect = TfidfVectorizer()
        TFIDF_vect.fit(corpus)

        with open('./model/TFIDF.pickle', "wb") as file:
            pickle.dump(TFIDF_vect, file)

    elif mode == "test":
        with open('./model/TFIDF.pickle', "rb") as file:
            TFIDF_vect = pickle.load(file)

    else:
        print("Invalid mode selection")
        exit(0)

    # Generate TFIDF representations for out dataset
    tfidf_sentecnce1 = TFIDF_vect.transform(corpus_sentence1)
    tfidf_sentecnce2 = TFIDF_vect.transform(corpus_sentence2)

    # Different methods to generate features from TFIDF representations. Uncomment to use.
    """
    # Option 1: Element wise multiplication of TFIDF vectors for sentence1 and sentence2
    tfidf_feature_array = scipy.sparse.csc_matrix.multiply(tfidf_sentecnce1, tfidf_sentecnce2)
    """

    """
    # Option 2: Euclidean dot product between TFIDF vectors for sentence1 and sentence2
    tfidf_feature_array = scipy.sparse.csr_matrix.sum(scipy.sparse.csc_matrix.multiply(tfidf_sentecnce1, tfidf_sentecnce2), axis=1)
    """

    """
    # Option 3: Euclidean distance between TFIDF vectors for sentence1 and sentence2
    tfidf_distance = tfidf_sentecnce1 - tfidf_sentecnce2
    tfidf_feature = [np.linalg.norm(tfidf_distance[ind].toarray()) for ind in range(tfidf_distance.shape[0])]
    tfidf_feature_array = np.asarray(tfidf_feature).reshape(-1, 1)
    """

    """
    # Option 4: TFIDF vectors for concatenated string of sentence1 and sentence2 
    tfidf_feature_array = TFIDF_vect.transform(corpus)
    """

    # Option 5: TFIDF vectors for sentence1 and sentence2 and concatenated
    tfidf_feature_array = scipy.sparse.hstack((tfidf_sentecnce1, tfidf_sentecnce2))

    return tfidf_feature_array, tfidf_labels
