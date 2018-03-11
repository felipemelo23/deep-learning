#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 09:36:38 2018

@author: felipe
"""
import keras
from keras import LSTM


class AttentionLSTM(LSTM):
    def __init__(self, output_dim, context, attn_activation='tanh', **kwargs):
        self.context = context
        self.attn_activation = attn_activation
        
        super(AttentionLSTM, self).__init__(output_dim, **kwargs)
        
    def build(self, input_shape):
        super(AttentionLSTM, self).build(input_shape)

        if hasattr(self.context, '_keras_shape'):
            attention_dim = self.context._keras_shape[1]
        else:
            raise Exception('Layer could not be build: No information about expected input shape.')
            
        ''' IF SOMETHING GOES WRONG... CHECK THE MATRIX W_h DIMENSIONS '''
        self.W_h = self.inner_init((self.output_dim, attention_dim), name='{}_W_h'.format(self.name))
        self.w_h = keras.zeros((self.output_dim,), name='{}_w_h'.format(self.name))
        
        self.U_h = self.inner_init((self.output_dim, self.output_dim), name='{}_U_h'.format(self.name))
        self.u_h = keras.zeros((self.output_dim,), name='{}_u_h'.format(self.name))
        
        self.W_o = self.inner_init((self.output_dim, self.output_dim), name='{}_W_o'.format(self.name))
        self.w_o = keras.zeros((self.output_dim,), name='{}_w_o'.format(self.name))
        
        self.trainable_weights += [self.W_h, self.U_h, self.W_o, self.w_h, self.u_h, self.w_o]
        
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        
    def step(self, x, states):
        h, [h, c] = super(AttentionLSTM, self).step(x, states)
        attention = states[4]
        
        
        
        