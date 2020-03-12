from utils.read_data import read_data
from utils.generate_meta_input import generate_meta_input
from TFIDF.logistic_regression_train import logistic_regression_train
from TFIDF.logistic_regression_test import logistic_regression_test

# To be run once to generate cleaned and tokenized sentences stored as pickle files
# generate_meta_input()

data = read_data()
train_data = data[:3]
test_data = data[3:]

logistic_regression_train(train_data)
logistic_regression_test(test_data)