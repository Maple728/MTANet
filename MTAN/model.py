#!/usr/bin/env python
"""
@author: peter.s
@project: MTANet
@time: 2019/6/25 8:41
@desc:
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from MTAN.layers.base_layer import EncoderLayer, DecoderLayer, AttentionLayer


class MTANet(keras.Model):

    def __init__(self, cell_units,
                 enc_input_shapes,
                 dec_input_shape,
                 output_shape,
                 ext_feature_dim,
                 rnn_type='gru',
                 rnn_dropout=0.0,
                 rnn_activation=tf.nn.tanh):

        super(MTANet, self).__init__(name='MTANet')

        assert dec_input_shape[-1] == output_shape[-1], 'The dimension of dec_input should be same with output.'

        self.cell_units = cell_units
        # list of [steps, dim]
        self.enc_shapes = [shape[-2:] for shape in enc_input_shapes]
        self.dec_shape = dec_input_shape[-2:]
        self.ext_feature_dim = ext_feature_dim
        self.n_enc = len(self.enc_shapes)
        self.horizon = output_shape[-2]
        self.target_dim = output_shape[-1]

        # construct layers
        self.enc_layers = [EncoderLayer(self.cell_units, rnn_type, rnn_dropout, rnn_activation)
                           for _ in range(self.n_enc)]
        self.attention_layers = [AttentionLayer(self.cell_units) for _ in range(self.n_enc)]
        self.dec_layer = DecoderLayer(self.cell_units, rnn_type, rnn_dropout, rnn_activation)
        self.ext_feat_embed_layer = layers.Dense(self.cell_units, activation=tf.nn.relu)
        self.output_layers = [layers.Dense(self.cell_units, activation=tf.nn.relu),
                              layers.Dense(self.target_dim)]

    def call(self, inputs):
        # assign inputs
        enc_fwd_inputs, \
        enc_back_inputs, \
        dec_input, ext_feat_input = inputs

        # reverse enc_back_inputs
        enc_back_inputs_rev = [tf.reverse(back_input, axis=1) for back_input in enc_back_inputs]

        # list of [batch_size, steps, dim]
        enc_fwd_inputs.append(enc_back_inputs_rev)

        enc_inputs = enc_fwd_inputs

        # executing encoders
        all_enc_outputs = []
        for enc_input, enc_layer in zip(enc_inputs, self.enc_layers):
            enc_output, _ = enc_layer(enc_input)
            all_enc_outputs.append(enc_output)

        # attention with decoder
        dec_outputs, last_dec_state, last_c_t = self.run_bahdanau_attention(all_enc_outputs, dec_input)
        # shape -> [batch_size, cell_units + n_enc * cell_units]
        state_concat = tf.concat([last_dec_state[0], last_c_t], axis=-1)

        # shape -> [batch_size, horizon, cell_units + n_enc * cell_units]
        ed_output = tf.tile(state_concat[:, None], [1, self.horizon, 1])
        # shape -> [batch_size, horizon, cell_units]
        ext_embed = self.ext_feat_embed_layer(ext_feat_input)

        # shape -> [batch_size, horizon, (n_enc + 2) * cell_units]
        x = tf.concat([ed_output, ext_embed], axis=-1)
        # generating prediction
        for output_layer in self.output_layers:
            x = output_layer(x)

        # shape -> [batch_size, horizon, output_dim]
        return x

    def run_bahdanau_attention(self, all_enc_outputs, dec_input):
        # init
        last_state = None
        c_t = tf.zeros_like()
        dec_outputs = []

        # steps * [batch_size, dec_input_dim]
        dec_input_list = tf.split(dec_input, axis=1)
        for dec_input_t in dec_input_list:
            # run decoder
            # shape -> [batch_size, 1, n_enc * cell_units + dec_input_dim]
            rnn_input = tf.concat([c_t[:, None], dec_input_t], axis=1)
            dec_output, last_state = self.dec_layer(rnn_input, last_state)
            # assign value
            last_state_s = tf.concat(last_state, axis=-1)
            dec_outputs.append(dec_output)

            # apply attention over all enc_outputs
            context_vectors = []
            for enc_outputs, attention_layer in zip(all_enc_outputs, self.attention_layers):
                # [batch_size, cell_units]
                c_t_i, _ = attention_layer(last_state_s, enc_outputs)
                context_vectors.append(c_t_i)
            # shape -> [batch_size, n_enc * cell_units]
            c_t = tf.concat(context_vectors, axis=-1)

        # shape -> [batch_size, steps, cell_units]
        dec_outputs = tf.concat(dec_outputs, axis=1)

        return dec_outputs, last_state, c_t
