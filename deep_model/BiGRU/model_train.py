# Imports
import tempfile
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, TimeDistributed
from tensorflow.keras.layers import BatchNormalization, Bidirectional, GRU, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from deep_model.BiGRU.preprocess import preprocess_traindata
from deep_model.BiGRU.parameters import *
from utils.plot import plot


# Trains the BiLSTM model using the data passed as argument
def BG_model_train(data):
    # Preprocess the data and obtain the embedding weight matrix
    train_data, embed_matrix = preprocess_traindata(data)

    # Define the embedding layer with the obtained weight matrix
    embedding = Embedding(input_dim=embed_matrix.shape[0],
                          output_dim=EMBED_HIDDEN_SIZE,
                          weights=[embed_matrix],
                          input_length=MAX_SEQ_LEN,
                          trainable=TRAIN_EMBED)

    # Define the bidirectional GRU layer
    BiGRU = Bidirectional(GRU(GRU_UNITS))

    # Add a time distributed translation layer for better performance
    # Time distributed layer applies the same Dense layer to each temporal slice of input
    translation = TimeDistributed(Dense(SENT_HIDDEN_SIZE, activation=ACTIVATION))

    # Define the input layers and its shapes for premise and hypothesis
    premise = Input(shape=(MAX_SEQ_LEN,), dtype='int32')
    hypothesis = Input(shape=(MAX_SEQ_LEN,), dtype='int32')

    # Embed the premise and hypothesis
    premise_embedded = embedding(premise)
    hypothesis_embedded = embedding(hypothesis)

    # Apply the translation layer
    premise_translated = translation(premise_embedded)
    hypothesis_translated = translation(hypothesis_embedded)

    # Apply the bidirectional GRU layer
    premise_BiGRU = BiGRU(premise_translated)
    hypothesis_BiGRU = BiGRU(hypothesis_translated)

    # Apply Batch normalization
    premise_normalized = BatchNormalization()(premise_BiGRU)
    hypothesis_normalized = BatchNormalization()(hypothesis_BiGRU)

    # Concatenate the normalized premise and hypothesis and apply a dropout layer
    train_input = concatenate([premise_normalized, hypothesis_normalized])
    train_input = Dropout(DROPOUT)(train_input)

    # Apply the (Dense layer, Dropout layer. Batch normalization layer) unit : 1
    train_input = Dense(2 * SENT_HIDDEN_SIZE, activation=ACTIVATION, kernel_regularizer=l2(L2))(train_input)
    train_input = Dropout(DROPOUT)(train_input)
    train_input = BatchNormalization()(train_input)

    # Apply the (Dense layer, Dropout layer. Batch normalization layer) unit : 2
    train_input = Dense(2 * SENT_HIDDEN_SIZE, activation=ACTIVATION, kernel_regularizer=l2(L2))(train_input)
    train_input = Dropout(DROPOUT)(train_input)
    train_input = BatchNormalization()(train_input)

    # Apply the (Dense layer, Dropout layer. Batch normalization layer) unit : 3
    train_input = Dense(2 * SENT_HIDDEN_SIZE, activation=ACTIVATION, kernel_regularizer=l2(L2))(train_input)
    train_input = Dropout(DROPOUT)(train_input)
    train_input = BatchNormalization()(train_input)

    # Define the output Dense layer
    prediction = Dense(CATEGORIES, activation='softmax')(train_input)

    # Define the complete model
    model = Model(inputs=[premise, hypothesis], outputs=prediction)

    # Choosing an optimizer
    optimizer = RMSprop(lr=LEARNING_RATE, rho=RHO, epsilon=EPSILON, decay=DECAY)

    # Compile the model and print out the model summary
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    print("Training model")

    # ReduceLROnPlateau callback to reduce learning rate when the validation accuracy plateaus
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                                patience=PATIENCE,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)

    # Early stopping callback to stop training if we are not making any positive progress
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=PATIENCE)

    # ModelCheckpoint callback to save the model with best performance
    # A temporary file is created to which the intermediate model weights are stored
    _, tmpfn = tempfile.mkstemp()
    model_checkpoint = ModelCheckpoint(tmpfn, save_best_only=True, save_weights_only=True)

    callbacks = [early_stopping, model_checkpoint, learning_rate_reduction]

    # Train the model
    history = model.fit(x=[train_data[0], train_data[1]],
                        y=train_data[2],
                        batch_size=BATCH_SIZE,
                        epochs=TRAINING_EPOCHS,
                        validation_split=VALIDATION_SPLIT,
                        callbacks=callbacks)

    # Restore the best found model during validation
    model.load_weights(tmpfn)

    """
    # Uncomment for generating plots.
    plot(history, "BiGRU")
    """

    # Save the model as h5 file
    model.save("./model/BiGRU.h5")