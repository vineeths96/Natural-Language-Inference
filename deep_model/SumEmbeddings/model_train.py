from deep_model.SumEmbeddings.preprocess import preprocess_traindata
from deep_model.SumEmbeddings.parameters import *
from utils.plot import plot

import tempfile

import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import concatenate, LSTM, GRU, Dense, Input, Dropout, TimeDistributed
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Bidirectional, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop

from tensorflow.keras.regularizers import l2


def SE_model_train(data):
    train_data, embed_matrix = preprocess_traindata(data)

    embedding = Embedding(input_dim=embed_matrix.shape[0],
                          output_dim=EMBED_HIDDEN_SIZE,
                          weights=[embed_matrix],
                          input_length=MAX_SEQ_LEN,
                          trainable=TRAIN_EMBED)

    SumEmbeddings = Lambda(lambda data: K.sum(data, axis=1), output_shape=(SENT_HIDDEN_SIZE,))
    translation = TimeDistributed(Dense(SENT_HIDDEN_SIZE, activation=ACTIVATION))

    premise = Input(shape=(MAX_SEQ_LEN,), dtype='int32')
    hypothesis = Input(shape=(MAX_SEQ_LEN,), dtype='int32')

    premise_embedded = embedding(premise)
    hypothesis_embedded = embedding(hypothesis)

    premise_translated = translation(premise_embedded)
    hypothesis_translated = translation(hypothesis_embedded)

    premise_SumEmbed = SumEmbeddings(premise_translated)
    hypothesis_SumEmbed = SumEmbeddings(hypothesis_translated)
    premise_normalized = BatchNormalization()(premise_SumEmbed)
    hypothesis_normalized = BatchNormalization()(hypothesis_SumEmbed)

    train_input = concatenate([premise_normalized, hypothesis_normalized])
    train_input = Dropout(DROPOUT)(train_input)

    train_input = Dense(2 * SENT_HIDDEN_SIZE, activation=ACTIVATION, kernel_regularizer=l2(L2))(train_input)
    train_input = Dropout(DROPOUT)(train_input)
    train_input = BatchNormalization()(train_input)

    train_input = Dense(2 * SENT_HIDDEN_SIZE, activation=ACTIVATION, kernel_regularizer=l2(L2))(train_input)
    train_input = Dropout(DROPOUT)(train_input)
    train_input = BatchNormalization()(train_input)

    train_input = Dense(2 * SENT_HIDDEN_SIZE, activation=ACTIVATION, kernel_regularizer=l2(L2))(train_input)
    train_input = Dropout(DROPOUT)(train_input)
    train_input = BatchNormalization()(train_input)

    prediction = Dense(CATEGORIES, activation='softmax')(train_input)

    model = Model(inputs=[premise, hypothesis], outputs=prediction)

    optimizer = RMSprop(lr=LEARNING_RATE, rho=RHO, epsilon=EPISLON, decay=DECAY)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                                patience=3,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    print("Training model")
    _, tmpfn = tempfile.mkstemp()

    # Save the best model during validation and bail out of training early if we're not improving
    callbacks = [EarlyStopping(monitor='val_loss', patience=PATIENCE),
                 ModelCheckpoint(tmpfn, save_best_only=True, save_weights_only=True),
                 learning_rate_reduction]

    history = model.fit(x=[train_data[0], train_data[1]],
                        y=train_data[2],
                        batch_size=BATCH_SIZE,
                        epochs=TRAINING_EPOCHS,
                        validation_split=VALIDATION_SPLIT,
                        callbacks=callbacks)

    # Restore the best found model during validation
    model.load_weights(tmpfn)

    # Uncomment for generating plots.
    plot(history, "SumEmbeddings")

    # Save the model as h5 file
    model.save("./model/SumEmbeddings.h5")