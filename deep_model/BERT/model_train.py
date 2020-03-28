# Imports
from transformers import *
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Input, GlobalAveragePooling1D
from deep_model.BERT.bert_input import get_inputs
from deep_model.BERT.preprocess import preprocess
from deep_model.BERT.parameters import *
from utils.plot import plot


# Adapted from tutorial by Muralidharan M at
# https://medium.com/analytics-vidhya/bert-in-keras-tensorflow-2-0-using-tfhub-huggingface-81c08c5f81d8
def BERT_model_train(data):
    # Process the training data to convert into required format
    train_data = preprocess(data)

    # Download/Initialize BERT tokenizer
    bert_tokenizer_transformer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True)

    # Convert input data into BERT acceptable format
    X_train = get_inputs(input=train_data, tokenizer=bert_tokenizer_transformer, maxlen=100)
    Y_train = train_data[2]

    # Define input layers
    token_inputs = Input(MAX_SEQUENCE_LENGTH, dtype=tf.int32, name='input_word_ids')
    mask_inputs = Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_masks')
    seg_inputs = Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='input_segments')

    # Download/Initialize BERT model
    bert_model = TFBertModel.from_pretrained('bert-base-cased')

    # Freeze BERT layers to speed up training
    for weights in bert_model.bert.weights:
        weights._trainable = False

    # Define output classification layer with BERT output
    bert_output, _ = bert_model([token_inputs, mask_inputs, seg_inputs])
    bert_output_pool = GlobalAveragePooling1D()(bert_output)
    bert_output_pool = Dense(DENSE_UNITS, activation='relu')(bert_output_pool)

    output = Dense(CATEGORIES, activation='sigmoid', name='output')(bert_output_pool)

    # Define the new BERT model with the new output layer
    bert_model_new = Model([token_inputs, mask_inputs, seg_inputs], output)
    bert_model_new.summary()

    # Define the optimizer and compile the model
    optimizer = Adam(learning_rate=LEARNING_RATE)
    bert_model_new.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

    # Fit the training data to our model
    history = bert_model_new.fit(x=X_train,
                                 y=Y_train,
                                 epochs=EPOCHS,
                                 batch_size=BATCH_SIZE)

    """
    # Uncomment for generating plots.
    plot(history, "BERT")
    """

    # Save the model as h5 file
    bert_model_new.save("./model/BERT", save_format='tf')


