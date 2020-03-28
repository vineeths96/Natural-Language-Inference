# Imports
import pickle
import numpy as np
from deep_model.BiGRU.parameters import *

# Specify the path to GloVe files
GLOVE_DIR = f'./input/embeddings/glove.6B.{EMBEDDING_DIM}d.txt'


# Creates and returns a GloVe dictionary with word-vector key-value pair
def glove_dict():
    embedding_dict = {}

    # Open the GloVe embedding file
    file = open(GLOVE_DIR)

    for line in file:
        # Spilt the word and its embedding vector
        line_list = line.split()
        word = line_list[0]
        embeddings = np.asarray(line_list[1:], dtype=float)

        # Store the word and its embedding vector in a dictionary
        embedding_dict[word] = embeddings

    file.close()

    # Store the dictionary as a pickle file to reduce thw overhead of loading
    with open('./input/embeddings/glove_dict.pickle', "wb") as file:
        pickle.dump(embedding_dict, file)

    return embedding_dict
