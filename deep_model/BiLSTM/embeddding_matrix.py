import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from deep_model.BiLSTM.glove_dict import glove_dict
from deep_model.BiLSTM.parameters import *


def embedding_matrix(corpus):
    try:
        with open('./input/embeddings/glove_dict.pickle', "rb") as file:
            glove_embedding = pickle.load(file)
    except:
        glove_embedding = glove_dict()

    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(corpus)

    with open('./model/tokenizer.pickle', "wb") as file:
        pickle.dump(tokenizer, file)

    word_index = tokenizer.word_index
    # print("Found %s unique tokens." %len(word_index))

    embed_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))

    for word, ind in word_index.items():
        embedding_vector = glove_embedding.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embed_matrix[ind] = embedding_vector

    return embed_matrix, tokenizer
