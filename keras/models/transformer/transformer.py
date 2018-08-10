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
from keras.layers import BatchNormalization, Wrapper
from keras.utils.generic_utils import func_dump
from keras.utils.generic_utils import func_load
from keras.utils.generic_utils import deserialize_keras_object
from keras.utils.generic_utils import has_arg
from keras.legacy import interfaces

from .attention import Attention, SelfAttention
from .ffn import FeedFowardNetwork

class PrePostProcessingWrapper(Wrapper):

    def __init__(self,
                 layer,
                 params,
                 **kwargs):
        super(PrePostProcessingWrapper, self).__init__(layer, **kwargs)

        self.layer = layer
        self.postprocess_dropout = params.layer_postprocess_dropout

        self.layer_norm = BatchNormalization(input_shape=(None, params.hidden_size))

    def call(self, inputs, **kwargs):
        if isinstance(inputs, (list, tuple)):
            inputs = inputs[0]

        # Preprocessing: apply layer normalization
        y = self.layer_norm(inputs)

        # Get layer output
        y = self.layer(y, **kwargs)

        # Postprocessing: apply dropout and residual connection
        if self.train:
            y = K.dropout(y, self.postprocess_dropout)

        return inputs + y


class EncoderStack(Layer):

    def __init__(self,
                 params,
                 train):
        super(EncoderStack, self).__init__()
        self.layers = []
        for _ in range(num_hidden_layers):
            # Create sublayers for each layer.
            self_attention_layer = Attention(params.hidden_size,
                                             params.num_heads,
                                             params.attention_dropout,
                                             train)
            feed_forward_network = FeedFowardNetwork(params.hidden_size,
                                                     params.filter_size,
                                                     params.relu_dropout,
                                                     train)

            self.layers.append([
                PrePostProcessingWrapper(self_attention_layer, params, train),
                PrePostProcessingWrapper(feed_forward_network, params, train)])

        # Create final layer normalization layer.
        self.output_normalization = BatchNormalization(input_shape=(None, params.hidden_size))

    def call(self, inputs):
        if isinstance(inputs, (list, tuple)):
            inputs = inputs[0]
        for n, layer in enumerate(self.layers):
            # Run inputs through the sublayers.
            self_attention_layer = layer[0]
            feed_forward_network = layer[1]

            y = self_attention_layer(inputs)
            y = feed_forward_network(y)

        return self.output_normalization(y)

    # TODO: add get_config/from_config



class DecoderStack(Layer):

    def __init__(self, params, train):
        super(DecoderStack, self).__init__()
        self.layers = []
        for _ in range(params.num_hidden_layers):
            self_attention_layer = SelfAttention(params.hidden_size,
                                                 params.num_heads,
                                                 params.attention_dropout,
                                                 train)
        enc_dec_attention_layer = Attention(params.hidden_size,
                                            params.num_heads,
                                            params.attention_dropout,
                                            train)
        feed_forward_network = FeedFowardNetwork(params.hidden_size,
                                                 params.filter_size,
                                                 params.relu_dropout,
                                                 train)

        self.layers.append([
            PrePostProcessingWrapper(self_attention_layer, params, train),
            PrePostProcessingWrapper(enc_dec_attention_layer, params, train),
            PrePostProcessingWrapper(feed_forward_network, params, train)])

        # Create final layer normalization layer.
        self.output_normalization = BatchNormalization(input_shape=(None, params.hidden_size))

    def call(self, inputs):
        assert isinstance(inputs, (list, tuple)) and len(inputs) == 2
        decoder_inputs = inputs[0]
        encoder_outputs = inputs[1]

        for n, layer in enumerate(self.layers):
            # Run inputs through the sublayers.
            self_attention_layer = layer[0]
            enc_dec_attention_layer = layer[1]
            feed_forward_network = layer[2]

            y = self_attention_layer(decoder_inputs)
            y = enc_dec_attention_layer(y, encoder_outputs)
            y = feed_forward_network(y)

        return self.output_normalization(y)

    # TODO: add get_config/from_config
