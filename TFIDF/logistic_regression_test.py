import pickle

from TFIDF.TFIDF_features import TFIDF_features


def logistic_regression_test(test_data):
    test_feature, test_label = TFIDF_features(test_data, "test")

    with open('./model/LR.pickle', "rb") as file:
        LR_model = pickle.load(file)

    pred_lables = LR_model.predict(test_feature)

    with open('./tfidf.txt', "w") as file:
        for item in pred_lables:
            if item == 0:
                file.write("contradiction\n")
            elif item == 1:
                file.write("neutral\n")
            else:
                file.write("contradiction\n")

    score = LR_model.score(test_feature, test_label) * 100

    print("The classification accuracy for Logistic regression with TF-IDF features is {}%.".format(score))
