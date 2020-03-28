# Imports
import pickle
from TFIDF.TFIDF_features import TFIDF_features


# Loads and tests the logistic regression model
def logistic_regression_test(test_data):
    # Obtain the TFIDF features
    test_feature, test_label = TFIDF_features(test_data, "test")

    # Loads the logistic regression model from pickle file
    with open('./model/LR.pickle', "rb") as file:
        LR_model = pickle.load(file)

    # Tests the logistic regression model
    pred_labels = LR_model.predict(test_feature)

    with open('./tfidf.txt', "w") as file:
        for item in pred_labels:
            if item == 0:
                file.write("contradiction\n")
            elif item == 1:
                file.write("neutral\n")
            elif item == 2:
                file.write("entailment\n")
            else:
                pass

    # Evaluate and print the results
    score = LR_model.score(test_feature, test_label) * 100
    print("The classification accuracy for Logistic regression with TF-IDF features is {:.2f}%.".format(score))
