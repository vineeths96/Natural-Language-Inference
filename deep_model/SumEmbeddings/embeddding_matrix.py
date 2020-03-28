# Imports
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from deep_model.SumEmbeddings.glove_dict import glove_dict
from deep_model.SumEmbeddings.parameters import *


# Creates and returns an embedding matrix which serves as the weights for the
# initial embedding layer in the model. Also returns a tokenizer which assigns
# integers to words from a defined word-integer table
def embedding_matrix(corpus):
    # Try to load GloVe embedding dictionary if it exists. If not, create one
    try:
        with open('./input/embeddings/glove_dict.pickle', "rb") as file:
            glove_embedding = pickle.load(file)
    except:
        glove_embedding = glove_dict()

    # Initialize and fit Keras tokenizer to convert words to integers
    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(corpus)

    # Save the tokenizer as a pickle file so that the same tokenizer (word-integer)
    # mapping can be used during testing time
    with open('./model/tokenizer.pickle', "wb") as file:
        pickle.dump(tokenizer, file)

    # Get an word-integer dictionary and use that to create an weight matrix
    # i-th column of weight matrix will have the vector of word with integer value i in dictionary
    word_index = tokenizer.word_index
    embed_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))

    for word, ind in word_index.items():
        # Get the embedding vector from GloVe dictionary, if available
        # Words not in the Glove would have the embedding matrix vector as full zeroes
        embedding_vector = glove_embedding.get(word)

        if embedding_vector is not None:
            embed_matrix[ind] = embedding_vector

    return embed_matrix, tokenizer
