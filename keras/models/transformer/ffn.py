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


class FeedFowardNetwork(Layer):

    def __init__(self,
                 hidden_size,
                 filter_size,
                 relu_dropout,
                 train):
        self.hidden_size = hidden_size
        self.filter_size = filter_size
        self.relu_dropout = relu_dropout
        self.train = train

        self.filter_dense_layer = Dense(filter_size,
                                        use_bias=True,
                                        activation='relu',
                                        name=self.name + "_filter_layer")
        self.output_dense_layer = Dense(hidden_size,
                                        use_bias=True,
                                        name=self.name + "_output_layer")

    # TODO: add padding support
    def call(self, inputs, **kwargs):
        if isinstance(inputs, (list, tuple)):
            inputs = inputs[0]

        batch_size = K.shape(inputs)[0]
        length = K.shape(inputs)[1]

        output = self.filter_dense_layer(inputs)
        if self.train:
            output = K.dropout(output, self.relu_dropout)
        output = self.output_dense_layer(output)
        return output

    # TODO: add get_config/from_config