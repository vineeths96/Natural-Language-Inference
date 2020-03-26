from utils.read_data import read_data
# from utils.generate_meta_input import generate_meta_input
from TFIDF.logistic_regression_train import logistic_regression_train
from TFIDF.logistic_regression_test import logistic_regression_test

from deep_model.SumEmbeddings.model_train import SE_model_train
from deep_model.SumEmbeddings.model_test import SE_model_test

from deep_model.BiLSTM.model_train import BL_model_train
from deep_model.BiLSTM.model_test import BL_model_test


from deep_model.BiGRU.model_train import BG_model_train
from deep_model.BiGRU.model_test import BG_model_test

# To be run once to generate cleaned and tokenized sentences stored as pickle files
# generate_meta_input()

data = read_data()
train_data = data[:3]
test_data = data[3:]

""" DONOT DELTE
# TODO
data edit tfidf
Sumembeddings output save to root default to SE


logistic_regression_train(train_data)
logistic_regression_test(test_data)


SE_model_train(train_data)
SE_model_test(test_data)

BL_model_train(train_data)
BL_model_test(test_data)

BG_model_train(train_data)
BG_model_test(test_data)

"""

