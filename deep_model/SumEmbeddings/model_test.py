from deep_model.SumEmbeddings.preprocess import preprocess_testdata
from deep_model.SumEmbeddings.parameters import *
import numpy as np
from tensorflow.keras.models import load_model

# Uncomment for generating plots. Requires some libraries (see below)

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from utils.plot_confusion_matrix import plot_confusion_matrix
from tensorflow.keras.utils import plot_model


def SE_model_test(data):
    test_data = preprocess_testdata(data)

    output_file = open('./results/SumEmbeddings/SumEmbeddings.txt', 'w')

    # Check if model exists at 'model' directory
    try:
        model = load_model('./model/SumEmbeddings.h5')
    except:
        print("Trained model does not exist. Please train the model.\n")
        exit()

    # Obtain the predicted classes
    loss, accuracy = model.evaluate(x=[test_data[0], test_data[1]], y=test_data[2], batch_size=BATCH_SIZE)
    print(f"Test Loss: {loss}, Test Accuracy: {accuracy}\n")

    Y_pred = model.predict([test_data[0], test_data[1]])
    Y_pred = np.argmax(Y_pred, axis=1)
    Y_test = np.argmax(test_data[2], axis=1)

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

    # Uncomment for generating plots.

    confusion_mtx = confusion_matrix(Y_test, Y_pred)
    plot_confusion_matrix(confusion_mtx, "SumEmbeddings", classes=range(3))

    target_names = ["Class {}".format(i) for i in range(CATEGORIES)]
    classification_rep = classification_report(Y_test, Y_pred, target_names=target_names, output_dict=True)

    plt.figure()
    sns.heatmap(pd.DataFrame(classification_rep).iloc[:-1, :].T, annot=True)
    plt.savefig('./results/SumEmbeddings/classification_report.png')
    plt.show()
    plot_model(model, to_file='./results/SumEmbeddings/model_plot.png', show_shapes=True, show_layer_names=True)
