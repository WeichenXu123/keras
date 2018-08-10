from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import copy
import types as python_types
import warnings

from keras import backend as K
from keras.engine.base_layer import InputSpec
from keras.engine.base_layer import Layer
from keras.layers import Dense
from keras.utils.generic_utils import func_dump
from keras.utils.generic_utils import func_load
from keras.utils.generic_utils import deserialize_keras_object
from keras.utils.generic_utils import has_arg
from keras.legacy import interfaces


class Attention(Layer):

    def __init__(self,
                 hidden_size,
                 num_heads,
                 attention_dropout,
                 train,
                 **kwargs):
        if hidden_size % num_heads != 0:
            raise ValueError("Hidden size must be evenly divisible by the number of "
                             "heads.")

        super(Attention, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.train = train

        # Layers for linearly projecting the queries, keys, and values.
        self.q_dense_layer = Dense(hidden_size, use_bias=False, name=self.name + "_q")
        self.k_dense_layer = Dense(hidden_size, use_bias=False, name=self.name + "_k")
        self.v_dense_layer = Dense(hidden_size, use_bias=False, name=self.name + "_v")

        # self.input_spec = [InputSpec(shape=(None, hidden_size)),
        #                   InputSpec(shape=(None, hidden_size))]
        self.output_dense_layer = Dense(hidden_size, use_bias=False,
                                                  name=self.name + "_output_transform")

    def split_heads(self, x):
        batch_size = K.shape(x)[0]
        length = K.shape(x)[1]

        # Calculate depth of last dimension after it has been split.
        depth = (self.hidden_size // self.num_heads)

        # Split the last dimension
        x = K.reshape(x, (batch_size, length, self.num_heads, depth))

        # Transpose the result
        return K.permute_dimensions(x, (0, 2, 1, 3))

    def combine_heads(self, x):
        batch_size = K.shape(x)[0]
        length = K.shape(x)[2]
        x = K.permute_dimensions(x, (0, 2, 1, 3))  # --> [batch, length, num_heads, depth]
        return K.reshape(x, (batch_size, length, self.hidden_size))

    def call(self, inputs, **kwargs):
        assert isinstance(inputs, (list, tuple)) and len(inputs) == 2
        x = inputs[0]
        y = inputs[1]
        q = self.q_dense_layer(x)
        k = self.k_dense_layer(y)
        v = self.v_dense_layer(y)

        # Split q, k, v into heads.
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # Scale q to prevent the dot product between q and k from growing too large.
        depth = (self.hidden_size // self.num_heads)
        q *= depth ** -0.5

        # Calculate dot product attention
        logits = K.dot(q, K.transpose(k))
        weights = K.softmax(logits, axis=-1)
        if self.train:
            weights = K.dropout(weights, self.attention_dropout)
        attention_output = K.dot(weights, v)

        # Recombine heads --> [batch_size, length, hidden_size]
        attention_output = self.combine_heads(attention_output)

        # Run the combined outputs through another linear projection layer.
        attention_output = self.output_dense_layer(attention_output)
        return attention_output

    # TODO: add get_config/from_config

class SelfAttention(Attention):
    """Multiheaded self-attention layer."""

    def call(self, inputs):
        if isinstance(inputs, (list, tuple)):
            inputs = inputs[0]
        return super(SelfAttention, self).call([inputs, inputs])
