import tensorflow as tf
import numpy as np

class LSTMcell(object):
    def __init__(self, incoming, D_input, D_cell, weight_initializer, f_bias=1.0,
                 L2=False, h_act=tf.tanh, hidden_init=None, cell_init=None):
        """
        :param incoming: used for accepting input data, shape=[n_steps, n_samples, D_input]
        :param D_input: scalar, dim of input
        :param D_cell: scalar, dim of hidden state and memory cell
        :param weight_initializer: function for initializing weight params
        :param f_bias: scalar, bias for forget gate
        :param L2: bool
        :param h_act: activication function for hidden state
        :param hidden_init: initial hidden state
        :param cell_init: initial cell state
        """

        self.incoming = incoming
        self.D_input = D_input
        self.D_cell = D_cell
        self.weight_initializer = weight_initializer
        self.f_bias = f_bias
        self.h_act = h_act
        self.cell_init = cell_init
        self.hidden_init = hidden_init
        self.type = 'lstm'

        if hidden_init is None and cell_init is None:
            init_for_both = tf.matmul(self.incoming[0,:,:], tf.zeros([D_input, D_cell]))
            self.cell_init = init_for_both
            self.hidden_init = init_for_both
        self.previous = tf.stack([self.hidden_init, self.cell_init], axis=0)

        # init params for forget gate layer
        self.fgate = self.Gate(f_bias)

        # init params for input gate layer
        self.igate = self.Gate()
        # init params for cell state
        self.cgate = self.Gate()

        # init params for output gate layer
        self.ogate = self.Gate()

        self.W_x = tf.concat([self.fgate[0], self.igate[0], self.cgate[0], self.ogate[0]], axis=1);    #shape = [D_input, 4*D_cell]
        self.W_h = tf.concat([self.fgate[1], self.igate[1], self.cgate[1], self.ogate[1]], axis=1);    #shape = [D_cell, 4*D_cell]
        self.b = tf.concat([self.fgate[2], self.igate[2], self.cgate[2], self.ogate[2]], axis=0);    #shape=[4*D_cell]

        if L2:
            self.L2_loss = tf.nn.l2_loss(self.W_x) + tf.nn.l2_loss(self.W_h)

    def Gate(self, bias = 0.001):
        Wx = self.weight_initializer([self.D_input, self.D_cell])
        Wh = self.weight_initializer([self.D_cell, self.D_cell])
        b = tf.Variable(tf.constant(bias, shape=[self.D_cell]), trainable=True)
        return Wx, Wh, b

    def Slice(self, x, n):
        return x[:, n*self.D_cell:(n+1)*self.D_cell]

    def Step(self, previous_h_c_tuple, current_x):
        """
        :param previous_h_c_tuple: tuple of (previous_hidden_state, previous_cell_state)
        :param current_x: current input
        :return: tuple of (current_hidden_state, current_cell_state)
        """
        prev_hidden, prev_cell = tf.unstack(previous_h_c_tuple)

        # cal all gates
        gates = tf.matmul(current_x, self.W_x) + tf.matmul(prev_hidden, self.W_h) + self.b
        # cal forget gate
        f = tf.sigmoid(self.Slice(gates, 0))
        # cal input gate
        i = tf.sigmoid(self.Slice(gates, 1))
        # cal current cell
        c = tf.tanh(self.Slice(gates, 2))
        cur_cell = prev_cell*f + i*c
        # cal current hidden
        o = tf.sigmoid(self.Slice(gates, 3))
        cur_hidden = tf.tanh(cur_cell)*o

        return tf.stack([cur_hidden, cur_cell])

class GRUcell(object):
    def __init__(self, incoming, D_input, D_cell, weight_initializer, L2=False, hidden_init=None):
        """
        :param incoming: used for accepting input data, shape=[n_steps, n_samples,D_input]
        :param D_input: scalar, dim of input
        :param D_cell: scalar, dim of hidden state and memory cell
        :param weight_initializer: function for initializing weight params
        :param L2: bool
        :param hidden_init: initial hidden state
        """
        self.incoming = incoming
        self.D_input = D_input
        self.D_cell = D_cell
        self.weight_initializer = weight_initializer
        self.hidden_init = hidden_init
        self.type = 'gru'

        if hidden_init is None :
            self.hidden_init = tf.matmul(self.incoming[0,:,:], tf.zeros([D_input, D_cell]))
        self.previous = self.hidden_init

        # init params for reset gate layer
        self.rgate = self.Gate()

        # init params for update gate layer
        self.ugate = self.Gate()
        # init params for cell state
        self.cgate = self.Gate()

        self.W_x = tf.concat([self.rgate[0], self.ugate[0], self.cgate[0]], axis=1);    #shape = [D_input, 4*D_cell]
        self.W_h = tf.concat([self.rgate[1], self.ugate[1], self.cgate[1]], axis=1);    #shape = [D_cell, 4*D_cell]
        self.b = tf.concat([self.rgate[2], self.ugate[2], self.cgate[2]], axis=0);    #shape=[4*D_cell]

        if L2:
            self.L2_loss = tf.nn.l2_loss(self.W_x) + tf.nn.l2_loss(self.W_h)

    def Gate(self, bias = 0.001):
        Wx = self.weight_initializer([self.D_input, self.D_cell])
        Wh = self.weight_initializer([self.D_cell, self.D_cell])
        b = tf.Variable(tf.constant(bias, shape=[self.D_cell]), trainable=True)
        return Wx, Wh, b

    def Slice(self, x, n):
        return x[:, n*self.D_cell:(n+1)*self.D_cell]

    def Step(self, previous_h, current_x):
        """
        :param previous_h: previous_hidden_state
        :param current_x: current input
        :return:current_hidden_state
        """

        W_x = tf.matmul(current_x, self.W_x) + self.b
        W_h = tf.matmul(previous_h, self.W_h)
        # cal reset gate
        r = tf.sigmoid(self.Slice(W_x, 0) + self.Slice(W_h, 0))
        # cal update gate
        u = tf.sigmoid(self.Slice(W_x, 1) + self.Slice(W_h, 1))
        c = tf.tanh(self.Slice(W_x, 2) + r*self.Slice(W_h, 2))
        # cal current hidden
        current_h = u*c + (1-u)*previous_h

        return current_h
