import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import GRU, LSTM, Dense, TimeDistributed, Activation, Bidirectional, RepeatVector, \
    Flatten, Permute, Dropout, Lambda, Reshape, UpSampling1D, Input, Multiply, Dot, Add, Embedding, BatchNormalization, \
    Reshape, GlobalMaxPool1D
import crowd_layer.crowd_layers, crowd_layer.crowd_aggregators
from crowd_layer.crowd_layers import CrowdsClassification, MaskedMultiCrossEntropy
from crowd_layer.crowd_aggregators import CrowdsCategoricalAggregator
import BertHandler as BH
import config as C
from tensorflow.contrib.metrics import f1_score
import numpy as np


class F1score(object):

    def loss(self, y_true, y_pred):
         return f1_score(y_true, y_pred)


def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)


# convenience l2_norm function
def l2_norm(x, axis=None):
    """
    takes an input tensor and returns the l2 norm along specified axis
    """

    square_sum = K.sum(K.square(x), axis=axis, keepdims=True)
    norm = K.sqrt(K.maximum(square_sum, K.epsilon()))

    return norm


def cosine_sim(A_B):
    """
    A [batch x n x d] tensor of n rows with d dimensions
    B [batch x m x d] tensor of n rows with d dimensions

    returns:
    D [batch x n x m] tensor of cosine similarity scores between each point i<n, j<m
    """

    A, B = A_B
    A_mag = l2_norm(A, axis=2)
    B_mag = l2_norm(B, axis=2)
    num = K.batch_dot(A, K.permute_dimensions(B, (0,2,1)))
    den = (A_mag * K.permute_dimensions(B_mag, (0,2,1)))
    dist_mat = num / den

    return dist_mat

# def build_model(max_seq_length):
#     in_id = Input(shape=(max_seq_length,), name="input_ids")
#     in_mask = Input(shape=(max_seq_length,), name="input_masks")
#     in_segment = Input(shape=(max_seq_length,), name="segment_ids")
#     bert_inputs = [in_id, in_mask, in_segment]
#
#     bert_output = BH.BertLayer(n_fine_tune_layers=3)(bert_inputs)
#     dense = Dense(32, activation='relu')(bert_output)
#     pred = Dense(2, activation='softmax')(dense)
#
#     model = Model(inputs=bert_inputs, outputs=pred)
#     # model.compile(loss=F1score.loss(), optimizer='adam', metrics=['accuracy'])
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
#     return model


def build_model_bert_lstm(max_seq_length):
    in_idA = Input(shape=(max_seq_length,), name="input_idsA")
    in_maskA = Input(shape=(max_seq_length,), name="input_masksA")
    in_segmentA = Input(shape=(max_seq_length,), name="segment_idAs")
    bert_inputsA = [in_idA, in_maskA, in_segmentA]
    bert_outputA = BH.BertLayer(n_fine_tune_layers=3, name='bert_inputA')(bert_inputsA)

    in_idB = Input(shape=(max_seq_length,), name="input_idsB")
    in_maskB = Input(shape=(max_seq_length,), name="input_masksB")
    in_segmentB = Input(shape=(max_seq_length,), name="segment_idsB")
    bert_inputsB = [in_idB, in_maskB, in_segmentB]
    bert_outputB = BH.BertLayer(n_fine_tune_layers=3, name='bert_inputB')(bert_inputsB)

    # bert_output = Multiply()([bert_outputA, bert_outputB])
    bert_output = Add()([bert_outputA, bert_outputB])
    bert_output = Reshape((-1, 1))(bert_output)
    bert_lstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.1))(bert_output)
    bert_lstm = GlobalMaxPool1D()(bert_lstm)
    # bert_lstm = Dropout(0.2)(bert_lstm)
    dense = Dense(64, activation='relu')(bert_lstm)
    pred = Dense(2, activation='softmax')(dense)

    model = Model(inputs=[in_idA, in_maskA, in_segmentA, in_idB, in_maskB, in_segmentB], outputs=pred)
    # model.compile(loss=F1score.loss(), optimizer='adam', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print(model.summary())
    return model


def build_model_bert(max_seq_length):
    in_idA = Input(shape=(max_seq_length,), name="input_idsA")
    in_maskA = Input(shape=(max_seq_length,), name="input_masksA")
    in_segmentA = Input(shape=(max_seq_length,), name="segment_idAs")
    bert_inputsA = [in_idA, in_maskA, in_segmentA]
    bert_outputA = BH.BertLayer(n_fine_tune_layers=3, name='bert_inputA')(bert_inputsA)

    in_idB = Input(shape=(max_seq_length,), name="input_idsB")
    in_maskB = Input(shape=(max_seq_length,), name="input_masksB")
    in_segmentB = Input(shape=(max_seq_length,), name="segment_idsB")
    bert_inputsB = [in_idB, in_maskB, in_segmentB]
    bert_outputB = BH.BertLayer(n_fine_tune_layers=3, name='bert_inputB')(bert_inputsB)

    # bert_output = Multiply()([bert_outputA, bert_outputB])
    bert_output = Add()([bert_outputA, bert_outputB])
    dense = Dense(64, activation='relu')(bert_output)
    pred = Dense(2, activation='softmax')(dense)

    model = Model(inputs=[in_idA, in_maskA, in_segmentA, in_idB, in_maskB, in_segmentB], outputs=pred)
    # model.compile(loss=F1score.loss(), optimizer='adam', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print(model.summary())
    return model


def build_model_glove_lstm(vocab_size, embedding_matrix, max_seq_length):
    e_a = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=max_seq_length, trainable=False)
    e_b = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=max_seq_length, trainable=False)
    inputA = Input(shape=(max_seq_length,), dtype='int32')
    inputB = Input(shape=(max_seq_length,), dtype='int32')
    glove_outputA = e_a(inputA)
    glove_outputB = e_b(inputB)

    glove_outputA = Flatten()(glove_outputA)
    glove_outputB = Flatten()(glove_outputB)

    # bert_output = Multiply()([bert_outputA, bert_outputB])
    glove_output = Add()([glove_outputA, glove_outputB])
    glove_output = Reshape((max_seq_length, 300))(glove_output)
    glove_lstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.1))(glove_output)
    glove_lstm = GlobalMaxPool1D()(glove_lstm)
    # glove_lstm = Dropout(0.2)(glove_lstm)
    dense = Dense(64, activation='relu')(glove_lstm)
    pred = Dense(2, activation='softmax')(dense)

    model = Model(inputs=[inputA, inputB], outputs=pred)
    # model.compile(loss=F1score.loss(), optimizer='adam', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print(model.summary())
    return model


def build_model_glove(vocab_size, embedding_matrix, max_seq_length):
    e_a = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=max_seq_length, trainable=False)
    e_b = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=max_seq_length, trainable=False)
    inputA = Input(shape=(max_seq_length,), dtype='int32')
    inputB = Input(shape=(max_seq_length,), dtype='int32')
    glove_outputA = e_a(inputA)
    glove_outputB = e_b(inputB)

    glove_outputA = Flatten()(glove_outputA)
    glove_outputB = Flatten()(glove_outputB)

    # bert_output = Multiply()([bert_outputA, bert_outputB])
    glove_output = Add()([glove_outputA, glove_outputB])
    dense = Dense(64, activation='relu')(glove_output)
    pred = Dense(2, activation='softmax')(dense)

    model = Model(inputs=[inputA, inputB], outputs=pred)
    # model.compile(loss=F1score.loss(), optimizer='adam', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print(model.summary())
    return model


def bert_as_matcher(max_seq_length):
    in_idA = Input(shape=(max_seq_length,), name="input_idsA")
    in_maskA = Input(shape=(max_seq_length,), name="input_masksA")
    in_segmentA = Input(shape=(max_seq_length,), name="segment_idAs")
    bert_inputsA = [in_idA, in_maskA, in_segmentA]
    bert_outputA = BH.BertLayer(n_fine_tune_layers=0, name='bert_inputA')(bert_inputsA)

    in_idB = Input(shape=(max_seq_length,), name="input_idsB")
    in_maskB = Input(shape=(max_seq_length,), name="input_masksB")
    in_segmentB = Input(shape=(max_seq_length,), name="segment_idsB")
    bert_inputsB = [in_idB, in_maskB, in_segmentB]
    bert_outputB = BH.BertLayer(n_fine_tune_layers=0, name='bert_inputB')(bert_inputsB)

    pred = Dot(1, normalize=True)([bert_outputA, bert_outputB])

    # bert_output = BH.BertLayer(n_fine_tune_layers=3)(bert_inputs)
    # pred = Dense(2, activation='softmax')(bert_output)
    # pred = Lambda(lambda x: [1 - x, x])(bert_output)
    model = Model(inputs=[in_idA, in_maskA, in_segmentA,
                          in_idB, in_maskB, in_segmentB], outputs=[bert_outputA, bert_outputB])
    # loss = F1score().loss
    # model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print(model.summary())
    return model


def glove_as_matcher(vocab_size, embedding_matrix, max_seq_length):
    e_a = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=max_seq_length, trainable=False)
    e_b = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=max_seq_length, trainable=False)
    inputA = Input(shape=(max_seq_length,), dtype='int32')
    inputB = Input(shape=(max_seq_length,), dtype='int32')
    glove_outputA = e_a(inputA)
    glove_outputB = e_b(inputB)

    pred = Dot(1, normalize=True)([glove_outputA, glove_outputB])
    # pred = Lambda(cosine_sim)([glove_outputA, glove_outputB])
    # pred = Dense(2, activation='softmax')(pred)

    # model = Model(inputs=[inputA, inputB], outputs=pred)
    model = Model(inputs=[inputA, inputB], outputs=[glove_outputA, glove_outputB])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print(model.summary())
    return model


# Build crowd model
def build_crowd_model_bert_lstm(max_seq_length, N_CLASSES, N_ANNOT):
    in_idA = Input(shape=(max_seq_length,), name="input_idsA")
    in_maskA = Input(shape=(max_seq_length,), name="input_masksA")
    in_segmentA = Input(shape=(max_seq_length,), name="segment_idAs")
    bert_inputsA = [in_idA, in_maskA, in_segmentA]
    bert_outputA = BH.BertLayer(n_fine_tune_layers=3, name='bert_inputA')(bert_inputsA)

    in_idB = Input(shape=(max_seq_length,), name="input_idsB")
    in_maskB = Input(shape=(max_seq_length,), name="input_masksB")
    in_segmentB = Input(shape=(max_seq_length,), name="segment_idsB")
    bert_inputsB = [in_idB, in_maskB, in_segmentB]
    bert_outputB = BH.BertLayer(n_fine_tune_layers=3, name='bert_inputB')(bert_inputsB)

    # bert_output = Multiply()([bert_outputA, bert_outputB])
    bert_output = Add()([bert_outputA, bert_outputB])
    bert_output = Reshape((-1, 1))(bert_output)
    bert_lstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.1))(bert_output)
    bert_lstm = GlobalMaxPool1D()(bert_lstm)
    # bert_lstm = Dropout(0.2)(bert_lstm)
    dense = Dense(64, activation='relu')(bert_lstm)
    pred = Dense(2, activation='softmax')(dense)

    # add crowds layer on top of the base model
    crowd = CrowdsClassification(N_CLASSES, N_ANNOT, conn_type="MW")(pred)
    model = Model(inputs=[in_idA, in_maskA, in_segmentA, in_idB, in_maskB, in_segmentB], outputs=crowd)

    # instantiate specialized masked loss to handle missing answers
    loss = MaskedMultiCrossEntropy().loss

    # compile model with masked loss and train
    model.compile(optimizer='adam', loss=loss)
    # print(model.summary())

    return model


def build_crowd_model_bert(max_seq_length, N_CLASSES, N_ANNOT):
    in_idA = Input(shape=(max_seq_length,), name="input_idsA")
    in_maskA = Input(shape=(max_seq_length,), name="input_masksA")
    in_segmentA = Input(shape=(max_seq_length,), name="segment_idAs")
    bert_inputsA = [in_idA, in_maskA, in_segmentA]
    bert_outputA = BH.BertLayer(n_fine_tune_layers=3, name='bert_inputA')(bert_inputsA)

    in_idB = Input(shape=(max_seq_length,), name="input_idsB")
    in_maskB = Input(shape=(max_seq_length,), name="input_masksB")
    in_segmentB = Input(shape=(max_seq_length,), name="segment_idsB")
    bert_inputsB = [in_idB, in_maskB, in_segmentB]
    bert_outputB = BH.BertLayer(n_fine_tune_layers=3, name='bert_inputB')(bert_inputsB)

    # bert_output = Multiply()([bert_outputA, bert_outputB])
    bert_output = Add()([bert_outputA, bert_outputB])
    dense = Dense(64, activation='relu')(bert_output)
    pred = Dense(2, activation='softmax')(dense)

    # add crowds layer on top of the base model
    crowd = CrowdsClassification(N_CLASSES, N_ANNOT, conn_type="MW")(pred)
    model = Model(inputs=[in_idA, in_maskA, in_segmentA, in_idB, in_maskB, in_segmentB], outputs=crowd)

    # instantiate specialized masked loss to handle missing answers
    loss = MaskedMultiCrossEntropy().loss

    # compile model with masked loss and train
    model.compile(optimizer='adam', loss=loss)
    # print(model.summary())

    return model


def build_crowd_model_glove(vocab_size, embedding_matrix,max_seq_length, N_CLASSES, N_ANNOT):
    e_a = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=max_seq_length, trainable=False)
    e_b = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=max_seq_length, trainable=False)
    inputA = Input(shape=(max_seq_length,), dtype='int32')
    inputB = Input(shape=(max_seq_length,), dtype='int32')
    glove_outputA = e_a(inputA)
    glove_outputB = e_b(inputB)

    glove_outputA = Flatten()(glove_outputA)
    glove_outputB = Flatten()(glove_outputB)

    # bert_output = Multiply()([bert_outputA, bert_outputB])
    glove_output = Add()([glove_outputA, glove_outputB])
    # glove_lstm = LSTM(128, 300*max_seq_length,
    #                   return_sequences=True,
    #                   dropout=0.25,
    #                   recurrent_dropout=0.1)(glove_output)
    dense = Dense(64, activation='relu')(glove_output)
    pred = Dense(2, activation='softmax')(dense)

    # add crowds layer on top of the base model
    crowd = CrowdsClassification(N_CLASSES, N_ANNOT, conn_type="MW")(pred)
    model = Model(inputs=[inputA, inputB], outputs=crowd)

    # instantiate specialized masked loss to handle missing answers
    loss = MaskedMultiCrossEntropy().loss

    # compile model with masked loss and train
    model.compile(optimizer='adam', loss=loss)
    # print(model.summary())

    return model


def build_crowd_model_glove_lstm(vocab_size, embedding_matrix, max_seq_length, N_CLASSES, N_ANNOT):
    e_a = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=max_seq_length, trainable=False)
    e_b = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=max_seq_length, trainable=False)
    inputA = Input(shape=(max_seq_length,), dtype='int32')
    inputB = Input(shape=(max_seq_length,), dtype='int32')
    glove_outputA = e_a(inputA)
    glove_outputB = e_b(inputB)

    glove_outputA = Flatten()(glove_outputA)
    glove_outputB = Flatten()(glove_outputB)

    # bert_output = Multiply()([bert_outputA, bert_outputB])
    glove_output = Add()([glove_outputA, glove_outputB])
    glove_output = Reshape((max_seq_length, 300))(glove_output)
    glove_lstm = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.1))(glove_output)
    glove_lstm = GlobalMaxPool1D()(glove_lstm)
    # glove_lstm = Dropout(0.2)(glove_lstm)
    dense = Dense(64, activation='relu')(glove_lstm)
    pred = Dense(2, activation='softmax')(dense)

    # add crowds layer on top of the base model
    crowd = CrowdsClassification(N_CLASSES, N_ANNOT, conn_type="MW")(pred)
    model = Model(inputs=[inputA, inputB], outputs=crowd)

    # instantiate specialized masked loss to handle missing answers
    loss = MaskedMultiCrossEntropy().loss

    # compile model with masked loss and train
    model.compile(optimizer='adam', loss=loss)
    # print(model.summary())

    return model


def remove_last_layer(model):
    model.layers.pop()
    model.layers.pop()
    model2 = Model(model.input, model.layers[-2].output)
    # model2.compile(optimizer='adam', loss=F1score.loss(), metrics=['accuracy'])
    model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print(model2.summary())
    return model2


def load_glove(file):
    embeddings_index = dict()
    f = open(file, encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index


def createEmbeddingMatrix(embeddings_index, vocab_size, t):
    embedding_matrix = np.zeros((vocab_size, 300))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
