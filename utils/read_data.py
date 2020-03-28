import pickle


# Reads pickled lists containing cleaned data (stored as lists of tokens)
def read_data():
    with open('./input/data_pickles/train_list_sentence1.txt', "rb") as file:
        train_list_sentence1 = pickle.load(file)

    with open('./input/data_pickles/train_list_sentence2.txt', "rb") as file:
        train_list_sentence2 = pickle.load(file)

    with open('./input/data_pickles/train_list_gold_label.txt', "rb") as file:
        train_list_gold_label = pickle.load(file)

    with open('./input/data_pickles/test_list_sentence1.txt', "rb") as file:
        test_list_sentence1 = pickle.load(file)

    with open('./input/data_pickles/test_list_sentence2.txt', "rb") as file:
        test_list_sentence2 = pickle.load(file)

    with open('./input/data_pickles/test_list_gold_label.txt', "rb") as file:
        test_list_gold_label = pickle.load(file)

    data = [[train_list_sentence1], [train_list_sentence2], [train_list_gold_label],
            [test_list_sentence1], [test_list_sentence2], [test_list_gold_label]]

    return data
