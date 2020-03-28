# Imports
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

import argparse
from utils.read_data import read_data
from utils.generate_meta_input import generate_meta_input
from TFIDF.logistic_regression_train import logistic_regression_train
from TFIDF.logistic_regression_test import logistic_regression_test
from deep_model.SumEmbeddings.model_train import SE_model_train
from deep_model.SumEmbeddings.model_test import SE_model_test
from deep_model.BiLSTM.model_train import BL_model_train
from deep_model.BiLSTM.model_test import BL_model_test
from deep_model.BiGRU.model_train import BG_model_train
from deep_model.BiGRU.model_test import BG_model_test
from deep_model.BERT.model_train import BERT_model_train
from deep_model.BERT.model_test import BERT_model_test

"""
# To be run once to generate cleaned and tokenized sentences stored as pickle files.
# Takes a significant amount of time to execute
generate_meta_input()
"""

# Reads the data from from pickle files
data = read_data()
train_data = data[:3]
test_data = data[3:]

# Command line argument parser. Defaults to testing the SumEmbeddings model.
arg_parser = argparse.ArgumentParser(description="Choose between training the model or testing the model. "
                                                 "Choose the model architecture between SumEmbeddings, "
                                                 "BiLSTM, BiGRU and BERT")

arg_parser.add_argument("--train-model", action="store_true", default=False)
arg_parser.add_argument("--model-name", type=str, default="SumEmbeddings")

argObj = arg_parser.parse_args()

# If train argument is present, train the network, else test the network and record accuracy
if argObj.train_model:
    logistic_regression_train(train_data)

    # Get model name and choose that model
    model_name = argObj.model_name
    if model_name == "SumEmbeddings":
        SE_model_train(train_data)
    elif model_name == "BiGRU":
        BG_model_train(train_data)
    elif model_name == "BiLSTM":
        BL_model_train(train_data)
    elif model_name == "BERT":
        BERT_model_train(train_data)
    else:
        print("Model not available. Choose an available model (Refer readme).")
else:
    logistic_regression_test(test_data)

    # Get model name and choose that model
    model_name = argObj.model_name
    if model_name == "SumEmbeddings":
        SE_model_test(test_data)
    elif model_name == "BiGRU":
        BG_model_test(test_data)
    elif model_name == "BiLSTM":
        BL_model_test(test_data)
    elif model_name == "BERT":
        BERT_model_test(test_data)
    else:
        print("Model not available. Choose an available model (Refer readme).")
