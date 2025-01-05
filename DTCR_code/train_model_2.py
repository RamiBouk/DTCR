
import tensorflow as tf
import numpy as np
import utils
import drnn
import rnn_cell_extensions


class Config:
    """Train config."""
    def __init__(self):
        self.batch_size = None
        self.hidden_size = [100, 50, 50]
        self.dilations = [1, 2, 4]
        self.num_steps = None
        self.embedding_size = None
        self.learning_rate = 1e-4
        self.cell_type = 'GRU'
        self.lamda = 1
        self.class_num = None
        self.denoising = True
        self.sample_loss = True


class RNNClusteringModel(tf.keras.Model):
    def __init__(self, config):
        super(RNNClusteringModel, self).__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.cell_type = config.cell_type
        self.denoising = config.denoising
        
        self.F = tf.Variable(tf.eye(config.batch_size, num_columns=config.class_num), trainable=False, dtype=tf.float32)

        if config.cell_type == 'GRU':
            self.rnn_cell = tf.keras.layers.GRUCell(np.sum(config.hidden_size) * 2)
        else:
            raise ValueError('Only GRU cell is supported.')

        self.decoder_wrapper = rnn_cell_extensions.LinearSpaceDecoderWrapper(self.rnn_cell, config.embedding_size)

        self.weight1 = tf.keras.layers.Dense(128, activation='relu')
        self.weight2 = tf.keras.layers.Dense(2)

    def call(self, inputs, noise, real_fake_label, F_new_value):
        if len(inputs.shape) != 3:
            raise ValueError(f"Expected inputs to have 3 dimensions [batch_size, num_steps, embedding_size], but got {inputs.shape}")

        batch_size, num_steps, embedding_size = inputs.shape

        if self.denoising:
            noise_input = inputs + noise
        else:
            noise_input = inputs

        reverse_noise_input = tf.reverse(noise_input, axis=[1])

        # Encoder outputs
        encoder_output_fw = drnn.drnn_layer_final(noise_input, self.hidden_size, self.config.dilations,
                                                  self.config.num_steps, embedding_size, self.cell_type)[1]

        encoder_output_bw = drnn.drnn_layer_final(reverse_noise_input, self.hidden_size, self.config.dilations,
                                                  self.config.num_steps, embedding_size, self.cell_type)[1]

        fw_states = [out[:, -1, :] for out in encoder_output_fw]
        bw_states = [out[:, -1, :] for out in encoder_output_bw]

        encoder_state_fw = tf.concat(fw_states, axis=1)
        encoder_state_bw = tf.concat(bw_states, axis=1)
        encoder_state = tf.concat([encoder_state_fw, encoder_state_bw], axis=1)

        # Decoder outputs
        decoder_inputs = utils._rnn_reformat(noise_input, input_dims=embedding_size, n_steps=self.config.num_steps)
        decoder_outputs, _ = tf.keras.layers.RNN(self.decoder_wrapper)(decoder_inputs, initial_state=encoder_state)

        # Loss components
        hidden_abstract = encoder_state
        self.F.assign(F_new_value)
        
        W = tf.transpose(hidden_abstract)
        WTW = tf.matmul(hidden_abstract, W)
        FTWTWF = tf.matmul(tf.matmul(tf.transpose(self.F), WTW), self.F)
        
        loss_reconstruct = tf.reduce_mean(tf.keras.losses.MeanSquaredError()(decoder_outputs, inputs))
        loss_k_means = tf.linalg.trace(WTW) - tf.linalg.trace(FTWTWF)

        hidden = self.weight1(hidden_abstract)
        output = self.weight2(hidden)
        discriminative_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=real_fake_label))

        total_loss = loss_reconstruct + self.config.lamda / 2 * loss_k_means + discriminative_loss

        return total_loss, hidden_abstract


def run_model(train_data_filename, config):
    train_data, train_label = utils.load_data(train_data_filename)
    config.batch_size = train_data.shape[0]
    config.num_steps = train_data.shape[1]
    config.embedding_size = 1

    train_label, num_classes = utils.transfer_labels(train_label)
    config.class_num = num_classes

    model = RNNClusteringModel(config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)

    for epoch in range(300):
        indices = np.random.permutation(train_data.shape[0])
        shuffle_data = train_data[indices]
        shuffle_label = train_label[indices]

        noise_data = np.random.normal(loc=0, scale=0.1, size=(shuffle_data.shape[0], shuffle_data.shape[1], 1))

        for input_batch, _ in utils.next_batch(config.batch_size, shuffle_data):
            print(f"input_batch shape: {input_batch.shape}")
            noise_batch = noise_data[:config.batch_size]
            fake_input, real_fake_label = utils.get_fake_sample(input_batch)

            with tf.GradientTape() as tape:
                loss_val, hidden_abstract = model(input_batch, noise_batch, real_fake_label, F_new_value=None)

            gradients = tape.gradient(loss_val, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            print(f"Epoch {epoch}, Loss: {loss_val.numpy()}")


def main():
    config = Config()
    filename = './Coffee/Coffee_TRAIN'
    run_model(filename, config)


if __name__ == "__main__":
    main()
