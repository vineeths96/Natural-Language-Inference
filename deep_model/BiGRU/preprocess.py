import pickle
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from deep_model.BiGRU.embeddding_matrix import embedding_matrix
from deep_model.BiGRU.parameters import *


def convert_data(data):
    list_sentence1 = data[0][0]
    list_sentence2 = data[1][0]
    list_gold_label = data[2][0]

    corpus_sentence1 = [' '.join(item) for item in list_sentence1]
    corpus_sentence2 = [' '.join(item) for item in list_sentence2]
    num_samples = len(list_gold_label)

    corpus = [corpus_sentence1[ind] + " " + corpus_sentence2[ind] for ind in range(num_samples)]

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

    del_list.sort(reverse=True)
    for ind in del_list:
        del corpus[ind]
        del corpus_sentence1[ind]
        del corpus_sentence2[ind]
        del labels[ind]

    labels = np.array(labels)
    data_converted = [corpus_sentence1, corpus_sentence2, labels]

    print(f"Non labelled: {len(del_list)}")
    print(f"Contradiction: {np.sum(labels == 0)}")
    print(f"Neutral: {np.sum(labels == 1)}")
    print(f"Entailment: {np.sum(labels == 2)}")

    return data_converted, corpus


def preprocess_traindata(train_data):
    print("Train Data information\n")
    data, corpus = convert_data(train_data)

    embed_matrix, tokenizer = embedding_matrix(corpus)

    sequence = lambda sentence: pad_sequences(tokenizer.texts_to_sequences(sentence), maxlen=MAX_SEQ_LEN)
    process = lambda item: (sequence(item[0]), sequence(item[1]), to_categorical(item[2]))

    training_data = process(data)

    return training_data, embed_matrix


def preprocess_testdata(test_data):
    print("Test Data information\n")
    data, _ = convert_data(test_data)

    with open('./model/tokenizer.pickle', "rb") as file:
        tokenizer = pickle.load(file)

    sequence = lambda sentence: pad_sequences(tokenizer.texts_to_sequences(sentence), maxlen=MAX_SEQ_LEN)
    process = lambda item: (sequence(item[0]), sequence(item[1]), to_categorical(item[2]))

    test_data = process(data)

    return test_data
