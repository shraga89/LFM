import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import GRU, LSTM, Dense, TimeDistributed, Activation, Bidirectional, RepeatVector, \
    Flatten, Permute, Dropout, Lambda, Reshape, UpSampling1D, Input, Multiply, Dot
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

def build_model(max_seq_length):
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

    bert_output = Multiply()([bert_outputA, bert_outputB])

    dense = Dense(32, activation='relu')(bert_output)
    pred = Dense(2, activation='softmax')(dense)

    model = Model(inputs=[in_idA, in_maskA, in_segmentA, in_idB, in_maskB, in_segmentB], outputs=pred)
    # model.compile(loss=F1score.loss(), optimizer='adam', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print(model.summary())
    return model


def bert_as_matcher(max_seq_length):
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

    bert_output = Dot(1, normalize=True)([bert_outputA, bert_outputB])

    # bert_output = BH.BertLayer(n_fine_tune_layers=3)(bert_inputs)
    pred = Dense(2, activation='softmax')(bert_output)

    model = Model(inputs=[in_idA, in_maskA, in_segmentA, in_idB, in_maskB, in_segmentB], outputs=pred)
    # loss = F1score().loss
    # model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # print(model.summary())
    return model


# Build crowd model
def build_crowd_model(max_seq_length, N_CLASSES, N_ANNOT):
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

    bert_output = Multiply()([bert_outputA, bert_outputB])
    dense = Dense(32, activation='relu')(bert_output)
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
    f = open(file)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
