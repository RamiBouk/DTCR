import itertools
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import RNN, SimpleRNNCell, GRUCell, LSTMCell

def dRNN(cell, inputs, rate, scope='default'):
    """
    Constructs a layer of dilated RNN.
    """
    n_steps = len(inputs)
    if rate < 0 or rate >= n_steps:
        raise ValueError('The \'rate\' variable needs to be adjusted.')

    print(f"Building layer: {scope}, input length: {n_steps}, dilation rate: {rate}, input dim: {inputs[0].shape[1]}")

    # Make the length of inputs divide 'rate' by using zero-padding
    EVEN = (n_steps % rate) == 0
    if not EVEN:
        zero_tensor = tf.zeros_like(inputs[0])
        dilated_n_steps = n_steps // rate + 1
        print(f"=====> {dilated_n_steps * rate - n_steps} time points need to be padded.")
        print(f"=====> Input length for sub-RNN: {dilated_n_steps}")
        for i_pad in range(dilated_n_steps * rate - n_steps):
            inputs.append(zero_tensor)
    else:
        dilated_n_steps = n_steps // rate
        print(f"=====> Input length for sub-RNN: {dilated_n_steps}")

    # Reshape inputs for dilated processing
    dilated_inputs = [tf.concat(inputs[i * rate:(i + 1) * rate], axis=0) for i in range(dilated_n_steps)]

    # Stack dilated inputs into a tensor
    dilated_inputs_tensor = tf.stack(dilated_inputs)

    # Use tf.keras.layers.RNN to process the dilated inputs
    rnn_layer = tf.keras.layers.RNN(cell, return_sequences=True)
    dilated_outputs = rnn_layer(dilated_inputs_tensor)

    # Unstack outputs to match the input format (list of tensors)
    unrolled_outputs = tf.unstack(dilated_outputs)
    outputs = unrolled_outputs[:n_steps]  # Remove padded zeros if any
    return outputs


def multi_dRNN_with_dilations(cells, inputs, dilations):
    """
    Constructs a multi-layer dilated RNN.
    """
    assert len(cells) == len(dilations)
    outputs = []
    x = inputs
    for i, (cell, dilation) in enumerate(zip(cells, dilations)):
        scope_name = f"multi_dRNN_dilation_{i}"
        x = dRNN(cell, x, dilation, scope=scope_name)
        outputs.append(x)
    return outputs


def _construct_cells(hidden_structs, cell_type):
    """
    Constructs a list of RNN cells.
    """
    cell_types = {"RNN": SimpleRNNCell, "LSTM": LSTMCell, "GRU": GRUCell}
    if cell_type not in cell_types:
        raise ValueError(f"The cell type '{cell_type}' is not supported.")
    return [cell_types[cell_type](hidden_dims) for hidden_dims in hidden_structs]


def _rnn_reformat(x, input_dims, n_steps):
    """
    Reformats input for standard RNN.
    """
    x_ = tf.transpose(x, [1, 0, 2])
    x_ = tf.reshape(x_, [-1, input_dims])
    x_reformat = tf.split(x_, n_steps, axis=0)
    return x_reformat


def drnn_layer_final(x, hidden_structs, dilations, n_steps, input_dims, cell_type):
    """
    Constructs a multilayer dilated RNN for classification.
    """
    assert len(hidden_structs) == len(dilations)

    # Reshape inputs
    x_reformat = _rnn_reformat(x, input_dims, n_steps)

    # Construct a list of cells
    cells = _construct_cells(hidden_structs, cell_type)

    # Define dRNN structures
    outputs = multi_dRNN_with_dilations(cells, x_reformat, dilations)

    return outputs
