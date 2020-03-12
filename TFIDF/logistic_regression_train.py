import pickle
from sklearn.linear_model import LogisticRegression

from TFIDF.TFIDF_features import TFIDF_features


def logistic_regression_train(data):
    train_feature, train_label = TFIDF_features(data, "train")

    LR_model = LogisticRegression(random_state=0, max_iter=1000)
    LR_model.fit(train_feature, train_label)

    with open('./model/LR.pickle', "wb") as file:
        pickle.dump(LR_model, file)
