import matplotlib.pyplot as plt


# Plot training performance
def plot(history, network):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='lower right')
    plt.tight_layout()
    plt.savefig('./results/' + network + '/accuracy_vs_epoch.png')
    # plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper right')
    plt.tight_layout()
    plt.savefig('./results/' + network + '/loss_vs_epoch.png')
    # plt.show()
