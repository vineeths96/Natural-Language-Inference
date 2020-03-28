# Imports
import pickle
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from deep_model.BiGRU.embeddding_matrix import embedding_matrix
from deep_model.BiGRU.parameters import *


# Process the data and return the data in required format
def convert_data(data):
    # Get the sentences and labels from composite data
    list_sentence1 = data[0][0]
    list_sentence2 = data[1][0]
    list_gold_label = data[2][0]

    # Merge each sublist (tokens list of each sentence) to a string
    corpus_sentence1 = [' '.join(item) for item in list_sentence1]
    corpus_sentence2 = [' '.join(item) for item in list_sentence2]
    num_samples = len(list_gold_label)

    # Create a composite corpus over which to train the Keras tokenizer
    # Corresponding lines of sentence1 and sentence2 are merged together
    corpus = [corpus_sentence1[ind] + " " + corpus_sentence2[ind] for ind in range(num_samples)]

    # There are entries in dataset without any gold_label (with gold_label entry as "-")
    # I choose to delete those from my data as they do not provide any information
    del_list = []
    labels = [None] * num_samples
    for ind, item in enumerate(list_gold_label):
        if item == "contradiction":
            labels[ind] = 0
        elif item == "neutral":
            labels[ind] = 1
        elif item == "entailment":
            labels[ind] = 2
        else:
            labels[ind] = 99
            del_list.append(ind)

    # Delete entries with gold_label "-"
    del_list.sort(reverse=True)
    for ind in del_list:
        del corpus[ind]
        del corpus_sentence1[ind]
        del corpus_sentence2[ind]
        del labels[ind]

    labels = np.array(labels)
    data_converted = [corpus_sentence1, corpus_sentence2, labels]

    """
    print(f"Non labelled: {len(del_list)}")
    print(f"Contradiction: {np.sum(labels == 0)}")
    print(f"Neutral: {np.sum(labels == 1)}")
    print(f"Entailment: {np.sum(labels == 2)}")
    """

    return data_converted, corpus


# Function to preprocess train data
def preprocess_traindata(train_data):
    # print("Train Data information\n")

    # Convert data to required format
    data, corpus = convert_data(train_data)

    # Obtain the embedding weight matrix and the tokenizer
    embed_matrix, tokenizer = embedding_matrix(corpus)

    # Process the data to integer sequences and labels to one-hot labels
    sequence = lambda sentence: pad_sequences(tokenizer.texts_to_sequences(sentence), maxlen=MAX_SEQ_LEN)
    process = lambda item: (sequence(item[0]), sequence(item[1]), to_categorical(item[2]))

    training_data = process(data)

    return training_data, embed_matrix


# Function to preprocess test data
def preprocess_testdata(test_data):
    # print("Test Data information\n")

    # Convert data to required format
    data, _ = convert_data(test_data)

    # Load the tokenizer from pickle file
    with open('./model/tokenizer.pickle', "rb") as file:
        tokenizer = pickle.load(file)

    # Process the data to integer sequences and labels to one-hot labels
    sequence = lambda sentence: pad_sequences(tokenizer.texts_to_sequences(sentence), maxlen=MAX_SEQ_LEN)
    process = lambda item: (sequence(item[0]), sequence(item[1]), to_categorical(item[2]))

    test_data = process(data)

    return test_data
