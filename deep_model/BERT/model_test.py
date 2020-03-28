# Imports
import numpy as np
from transformers import *
from tensorflow.keras.models import load_model
from deep_model.BERT.bert_input import get_inputs
from deep_model.BERT.preprocess import preprocess
from deep_model.BERT.parameters import *

"""
# Uncomment for generating plots. Requires some libraries (see below)
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from utils.plot_confusion_matrix import plot_confusion_matrix
from tensorflow.keras.utils import plot_model
"""


# Tests the BERT model using the data passed as argument
def BERT_model_test(data):
    test_data = preprocess(data)

    # Download/Initialize BERT tokenizer
    bert_tokenizer_transformer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True)

    # Convert input data into BERT acceptable format
    X_test = get_inputs(input=test_data, tokenizer=bert_tokenizer_transformer, maxlen=100)
    Y_test = test_data[2]

    # Open the file to which output has to be written
    output_file = open('./results/BERT/BERT.txt', 'w')

    # Check if model exists at 'model' directory
    try:
        bert_model = load_model('./model/BERT')
    except:
        print("Trained model does not exist. Please train the model.\n")
        exit(0)

    # Obtain the predicted classes
    Y_pred = bert_model.predict(X_test)
    Y_pred = np.argmax(Y_pred, axis=1)

    # Write output to file
    for ind in range(Y_pred.shape[0]):
        if Y_pred[ind] == 0:
            output_file.write("contradiction\n")
        elif Y_pred[ind] == 1:
            output_file.write("neutral\n")
        elif Y_pred[ind] == 2:
            output_file.write("entailment\n")
        else:
            pass

    output_file.close()

    """
    # Uncomment for generating plots.
    confusion_mtx = confusion_matrix(Y_test, Y_pred)
    plot_confusion_matrix(confusion_mtx, "BERT", classes=range(3))

    target_names = ["Class {}".format(i) for i in range(CATEGORIES)]
    classification_rep = classification_report(Y_test, Y_pred, target_names=target_names, output_dict=True)

    plt.figure()
    sns.heatmap(pd.DataFrame(classification_rep).iloc[:-1, :].T, annot=True)
    plt.savefig('./results/BERT/classification_report.png')
    plt.show()
    plot_model(model, to_file='./results/BERT/model_plot.png', show_shapes=True, show_layer_names=True)
    """
