import numpy as np


# Process the data and return the data in required format
def preprocess(data):
    # Get the sentences and labels from composite data
    list_sentence1 = data[0][0]
    list_sentence2 = data[1][0]
    list_gold_label = data[2][0]

    # Merge each sublist (tokens list of each sentence) to a string
    corpus_sentence1 = [' '.join(item) for item in list_sentence1]
    corpus_sentence2 = [' '.join(item) for item in list_sentence2]
    num_samples = len(list_gold_label)

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
        del labels[ind]
        del corpus_sentence1[ind]
        del corpus_sentence2[ind]

    labels = np.array(labels)

    """
    print(f"Non labelled: {len(del_list)}\n")
    print(f"Contradiction:{np.sum(labels == 0)}\n")
    print(f"Neutral:{np.sum(labels == 1)}\n")
    print(f"Entailment:{np.sum(labels == 2)}\n")
    """

    return corpus_sentence1, corpus_sentence2, labels
