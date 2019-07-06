#!/usr/bin/env python
"""
@author: peter.s
@project: METANet
@time: 2019/6/24 20:42
@desc:
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers


class AttentionLayer(keras.Model):

    def __init__(self, cell_units, activation=tf.nn.tanh, name=''):
        super(AttentionLayer, self).__init__(name=name)
        self.cell_units = cell_units
        self.activation = activation
        # build parameters
        self.W1 = layers.Dense(cell_units)
        self.W2 = layers.Dense(cell_units)
        self.V = layers.Dense(1)

    def call(self, target, sources):
        """

        :param target: [batch_size, hidden_size]
        :param sources: [batch_size, n_vectors, hidden_size]
        :return:
        """
        if target is None:
            attention_weights = None
            context_vector = tf.reduce_mean(sources, axis=1)
        else:
            # shape -> [batch_size, n_vector, 1]
            # scaled score for fast gradient
            score = self.V(self.activation(
                self.W1(target[:, None]) + self.W2(sources)
            )) / tf.sqrt(self.cell_units)

            attention_weights = tf.nn.softmax(score, axis=1)

            # shape -> [batch_size, 1]
            context_vector = tf.reduce_sum(attention_weights * sources, axis=1)

        return context_vector, attention_weights


def create_rnn_layer(cell_units, cell_type, recurrent_dropout, activation):
    if cell_type.lower() == 'rnn':
        rnn_layer = layers.SimpleRNN
    elif cell_type.lower() == 'gru':
        rnn_layer = layers.GRU
    elif cell_type.lower() == 'lstm':
        rnn_layer = layers.LSTM
    else:
        # custom rnn layer
        rnn_layer = layers.LSTM

    return rnn_layer(cell_units,
                     recurrent_dropout=recurrent_dropout,
                     activation=activation,
                     return_sequences=True,
                     return_state=True)


def rnn_dynamic_run(rnn_layer, x, initial_state=None):
    tmp_res = rnn_layer(x, initial_state=initial_state)
    # outputs - > [batch_size, T, hidden_size], last_state ([h_t] or [h_t, s_t])
    return tmp_res[0], tmp_res[1:]


class EncoderLayer(keras.Model):

    def __init__(self, cell_units, rnn_type, rnn_dropout, rnn_activation,
                 name='Encoder'):
        super(EncoderLayer, self).__init__(name=name)

        self.rnn_layer = create_rnn_layer(cell_units, rnn_type, rnn_dropout, rnn_activation)

    def call(self, x, initial_state=None):
        return rnn_dynamic_run(self.rnn_layer, x, initial_state=initial_state)


class DecoderLayer(keras.Model):

    def __init__(self, cell_units, rnn_type, rnn_dropout, rnn_activation,
                 name='Decoder'):
        super(DecoderLayer, self).__init__(name=name)

        self.rnn_layer = create_rnn_layer(cell_units, rnn_type, rnn_dropout, rnn_activation)

    def call(self, x, initial_state=None):
        return rnn_dynamic_run(self.rnn_layer, x, initial_state=initial_state)