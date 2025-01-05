import tensorflow as tf

class LinearSpaceDecoderWrapper(tf.keras.layers.Layer):
    """Operator adding a linear encoder to an RNN cell"""

    def __init__(self, cell, output_size):
        """Create a cell with a linear encoder in space.

        Args:
          cell: a tf.keras.layers.Layer (e.g., GRUCell or LSTMCell). Input is passed through a linear layer.
          output_size: int, the size of the output projection layer.

        Raises:
          TypeError: if cell is not a valid RNN layer.
        """
        super(LinearSpaceDecoderWrapper, self).__init__()

        if not isinstance(cell, tf.keras.layers.Layer):
            raise TypeError("The parameter cell is not a valid tf.keras layer.")

        self._cell = cell
        self.linear_output_size = output_size

        print(f'output_size = {output_size}')
        print(f'state_size = {self._cell.state_size}')

        # Initialize the linear projection layer
        self.w_out = self.add_weight(
            name="proj_w_out",
            shape=(self._cell.output_size, output_size),
            initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04),
        )

        self.b_out = self.add_weight(
            name="proj_b_out",
            shape=(output_size,),
            initializer=tf.random_uniform_initializer(minval=-0.04, maxval=0.04),
        )

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self.linear_output_size

    def call(self, inputs, states):
        """Use a linear layer and pass the output to the cell."""
        # Run the RNN as usual
        output, new_states = self._cell(inputs, states)

        # Apply the linear transformation
        output = tf.matmul(output, self.w_out) + self.b_out

        return output, new_states
