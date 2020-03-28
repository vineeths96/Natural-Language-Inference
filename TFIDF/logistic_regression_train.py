# Imports
import pickle
from sklearn.linear_model import LogisticRegression
from TFIDF.TFIDF_features import TFIDF_features


# Trains and stores a logistic regression model
def logistic_regression_train(train_data):
    # Obtain the TFIDF features
    train_feature, train_label = TFIDF_features(train_data, "train")

    # Train the logistic regression model
    LR_model = LogisticRegression(random_state=0, max_iter=1000, solver='lbfgs', multi_class='auto')
    LR_model.fit(train_feature, train_label)

    # Save the logistic regression model as a pickle file
    with open('./model/LR.pickle', "wb") as file:
        pickle.dump(LR_model, file)

    print("Training complete.\n")
