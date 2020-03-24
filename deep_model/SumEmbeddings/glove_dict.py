import pickle
import numpy as np
from deep_model.SumEmbeddings.parameters import *

GLOVE_DIR = f'./input/embeddings/glove.6B.{EMBEDDING_DIM}d.txt'


def glove_dict():
    embedding_dict = {}
    file = open(GLOVE_DIR)

    for line in file:
        line_list = line.split()
        word = line_list[0]
        embeddings = np.asarray(line_list[1:], dtype=float)

        embedding_dict[word] = embeddings

    with open('./input/embeddings/glove_dict.pickle', "wb") as file:
        pickle.dump(embedding_dict, file)

    file.close()

    return embedding_dict
