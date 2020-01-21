import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

import tensorflow.keras.layers as layers
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers

import DOT_preprocess

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True


class Encoder(tf.keras.layers.Layer):

    def __init__(self,
                 n_dims,
                 name='encoder',
                 **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.n_dims = n_dims
        self.n_layers = len(n_dims)
        init = tf.compat.v1.constant_initializer(0.1)
        self.encode_layer0 = layers.Dense(n_dims[0],
                                          activation='relu',
                                          kernel_regularizer=regularizers.l2(1e-4),
                                          bias_initializer='glorot_uniform')
        self.encode_layer1 = layers.Dense(n_dims[1],
                                          activation='relu',
                                          kernel_regularizer=regularizers.l2(1e-4),
                                          bias_initializer='glorot_uniform')
        self.encode_layer2 = layers.Dense(n_dims[2],
                                          activation='sigmoid',
                                          kernel_regularizer=regularizers.l2(1e-4),
                                          bias_initializer='glorot_uniform')

    @tf.function
    def call(self, inputs):
        x0 = self.encode_layer0(inputs)
        x1 = self.encode_layer1(x0)
        x2 = self.encode_layer2(x1)
        return x2


class Decoder(tf.keras.layers.Layer):

    def __init__(self,
                 n_dims,
                 name='decoder',
                 **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.n_dims = n_dims
        self.n_layers = len(n_dims)
        self.decode_layer0 = layers.Dense(n_dims[0],
                                          activation='relu',
                                          kernel_regularizer=regularizers.l2(1e-4),
                                          bias_initializer='glorot_uniform')
        self.decode_layer1 = layers.Dense(n_dims[1],
                                          activation='relu',
                                          kernel_regularizer=regularizers.l2(1e-4),
                                          bias_initializer='glorot_uniform')
        self.recon_layer = layers.Dense(n_dims[2],
                                        activation=None,
                                        kernel_regularizer=regularizers.l2(1e-4),
                                        bias_initializer='glorot_uniform')

    @tf.function
    def call(self, inputs):
        x0 = self.decode_layer0(inputs)
        x1 = self.decode_layer1(x0)
        y = self.recon_layer(x1)
        return y


class Autoencoder(tf.keras.Model):
    '''Vanilla Autoencoder for MNIST digits'''

    def __init__(self,
                 n_dims=[64, 128, 768, 256, 128, 252],
                 name='autoencoder',
                 **kwargs):
        super(Autoencoder, self).__init__(name=name, **kwargs)
        self.n_dims = n_dims
        self.encoder = Encoder(n_dims[:3])
        self.decoder = Decoder(n_dims[3:])

    @tf.function
    # def call(self, inputs):
    #     x = self.encoder(inputs)
    #     recon = self.decoder(x)
    #     return recon

    def encod(self, inputs):
        x = self.encoder(inputs)
        return x

    def decod(self, inputs):
        recon = self.decoder(inputs)
        return recon


def random_shuffle(x, y):
    perm = tf.random.shuffle(tf.range(tf.shape(x)[0]))
    x = tf.gather(x, perm, axis=0)
    y = tf.gather(y, perm, axis=0)
    return x, y


# def custom_loss(true, pred):
#     loss1 = tf.reduce_mean(tf.square(tf.subtract(pred, true)))
#     # loss2 = tf.add(loss1, l1l2(pred))
#     return loss1


training_epochs = 250
batch_size = 64
batch_size2 = 10
display_step = 1

ref_a, meas_simu, batch_ys_a, depth_a = DOT_preprocess.dataprocess()
ref_t, meas_phan, test_image = DOT_preprocess.test_data()

mua_true_distribution = batch_ys_a.reshape(-1, 256, 3)

# meas_simu = (meas_simu)/(np.max(meas_simu)-np.min(meas_simu))
# meas_phan = (meas_phan)/(np.max(meas_phan)-np.min(meas_phan))

batch_ys_a = batch_ys_a.reshape(-1, 768)

meas_simu = tf.cast(meas_simu, tf.float64)
meas_phan = tf.cast(meas_phan, tf.float64)
batch_ys_a = tf.cast(batch_ys_a, tf.float64)

train_data_simu, mua_true = random_shuffle(meas_simu, batch_ys_a)

optimizer_mea2mua_simu = tf.optimizers.Adam(learning_rate=1e-3)
optimizer_mua2mea_simu = tf.optimizers.Adam(learning_rate=1e-4)
optimizer_mea2mua_phan = tf.optimizers.Adam(learning_rate=1e-6)
optimizer_mua2mea_phan = tf.optimizers.Adam(learning_rate=1e-7)

l1 = tf.keras.regularizers.l1(l=1e-4)
l2 = tf.keras.regularizers.l2(l=1e-4)

mae_loss = tf.keras.losses.MeanAbsoluteError()
mse_loss = tf.keras.losses.MeanSquaredError()
loss_metric = tf.keras.metrics.Mean()

autoencoder = Autoencoder([64, 128, 768, 256, 128, 252])

loss_plot = []
# Iterate over epochs.
for epoch in range(training_epochs):
# for epoch in range(5):
    print(f'Epoch {epoch+1}')

    if epoch < 200:
  # Iterate over the batches of the dataset.
        for i in range(len(train_data_simu)//batch_size):
            train_data_simu_batch = train_data_simu[i*batch_size:(i+1)*batch_size]
            mua_true_batch = mua_true[i*batch_size:(i+1)*batch_size]

            with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
                # recon = autoencoder(x_batch)
                code = autoencoder.encod(train_data_simu_batch)
                recon = autoencoder.decod(mua_true_batch)

                loss1 = 1e0 * mae_loss(code, mua_true_batch) + \
                    0e-1 * tf.reduce_mean(tf.abs(tf.reduce_max(tf.cast(code, tf.float64), 1) - tf.reduce_max(tf.cast(mua_true_batch, tf.float64), 1)))
                loss2 = mae_loss(train_data_simu_batch, recon)
                loss_plot.append(loss1)

            grads1 = tape1.gradient(loss1, autoencoder.trainable_variables[:6])
            optimizer_mea2mua_simu.apply_gradients(zip(grads1, autoencoder.trainable_variables[:6]))
            loss_metric(loss1)

            grads2 = tape2.gradient(loss2, autoencoder.trainable_variables[6:])
            optimizer_mua2mea_simu.apply_gradients(zip(grads2, autoencoder.trainable_variables[6:]))
            # loss_metric(loss2)

            if i % 100 == 0:
                print(f'Step {i}: mean loss = {loss_metric.result()}')

    else:
        # for i in range(len(train_data_simu)//batch_size):
        for i in range(50):
            train_data_phan, _ = random_shuffle(meas_phan, meas_phan)
            train_data_phan_batch = train_data_phan[:10]

            with tf.GradientTape() as tape:
                # recon = autoencoder(train_data_phan_batch)
                recon1 = autoencoder.encod(train_data_phan_batch)
                recon = autoencoder.decod(recon1)
                loss = mse_loss(train_data_phan_batch, recon)

            grads = tape.gradient(loss, autoencoder.trainable_variables)
            optimizer_mea2mua_phan.apply_gradients(zip(grads[:6], autoencoder.trainable_variables[:6]))
            optimizer_mua2mea_phan.apply_gradients(zip(grads[6:], autoencoder.trainable_variables[6:]))
            loss_metric(loss)

            if i % 100 == 0:
                print(f'Step {i}: mean loss = {loss_metric.result()}')

result = autoencoder.encod(meas_phan).numpy().reshape(-1, 256, 3)
result2 = result.reshape(-1, 16, 16, 3)
a = result2[::-1, ::-1, ::-1, 1]
b = result2[::-1, ::-1, ::-1, 0]
c = result2[::-1, ::-1, ::-1, 2]

result1 = autoencoder.encod(meas_simu).numpy().reshape(-1, 256, 3)
plt.plot(loss_plot)
plt.show()


# train_data = tf.data.Dataset.from_tensor_slices(tf.cast(meas_simu, tf.float64))
# train_data = train_data.batch(batch_size)
# train_data = train_data.shuffle(meas_simu.shape[0])
# train_data = train_data.prefetch(batch_size*4)
#
# train_data2 = tf.data.Dataset.from_tensor_slices(tf.cast(meas_phan, tf.float64))
# train_data2 = train_data2.batch(batch_size2)
# train_data2 = train_data2.shuffle(meas_phan.shape[0])
# train_data2 = train_data2.prefetch(batch_size2*4)

# x = Autoencoder([512, 784, 256, 252]).encod()
# autoencoder = Autoencoder([512, 784, 256, 252]).decod()

# # Iterate over epochs.
# for epoch in range(5):
#     print(f'Epoch {epoch+1}')
#
#   # Iterate over the batches of the dataset.
#     for step, x_batch in enumerate(train_data):
#         with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
#           # recon = autoencoder(x_batch)
#           code = autoencoder.encod(x_batch)
#           recon = autoencoder.decod(code)
#
#           loss1 = mse_loss(code, mua_true)
#           loss2 = mse_loss(x_batch, recon)
#
#         grads1 = tape1.gradient(loss, autoencoder.trainable_variables[:3])
#         optimizer.apply_gradients(zip(grads1, autoencoder.trainable_variables[:3]))
#
#         grads2 = tape2.gradient(loss, autoencoder.trainable_variables[3:])
#         optimizer.apply_gradients(zip(grads2, autoencoder.trainable_variables[3:]))
#
#         loss_metric(loss)
#
#         if step % 100 == 0:
#           print(f'Step {step}: mean loss = {loss_metric.result()}')

# np.random.seed(1)
# tf.random.set_seed(1)
# batch_size = 64
# batch_size2 = 10
# epochs = 10
# learning_rate = 1e-2
# intermediate_dim = 784
# original_dim = 252

# class Encoder(tf.keras.layers.Layer):
#     def __init__(self, intermediate_dim):
#         super(Encoder, self).__init__()
#         self.hidden_layer = tf.keras.layers.Dense(
#             units=intermediate_dim,
#             activation=tf.nn.relu,
#             kernel_initializer='glorot_uniform'
#         )
#         self.output_layer = tf.keras.layers.Dense(
#             units=intermediate_dim,
#             activation=tf.nn.sigmoid
#         )
#
#     def call(self, input_features):
#         activation = self.hidden_layer(input_features)
#         return self.output_layer(activation)
#
#
# class Decoder(tf.keras.layers.Layer):
#     def __init__(self, intermediate_dim, original_dim):
#         super(Decoder, self).__init__()
#         self.hidden_layer = tf.keras.layers.Dense(
#             units=intermediate_dim,
#             activation=tf.nn.relu,
#             kernel_initializer='glorot_uniform'
#         )
#         self.output_layer = tf.keras.layers.Dense(
#             units=original_dim,
#             activation=tf.nn.sigmoid
#         )
#
#     def call(self, code):
#         activation = self.hidden_layer(code)
#         return self.output_layer(activation)
#
#
# class Autoencoder(tf.keras.Model):
#     def __init__(self, intermediate_dim, original_dim):
#         super(Autoencoder, self).__init__()
#         self.encoder = Encoder(intermediate_dim=intermediate_dim)
#         self.decoder = Decoder(
#             intermediate_dim=intermediate_dim,
#             original_dim=original_dim
#         )
#
#     def call(self, input_features):
#         code = self.encoder(input_features)
#         reconstructed = self.decoder(code)
#         return reconstructed
#
# autoencoder = Autoencoder(
#   intermediate_dim=intermediate_dim,
#   original_dim=original_dim
# )
# opt = tf.optimizers.Adam(learning_rate=learning_rate)
#
#
# def loss(model, original):
#     reconstruction_error = tf.reduce_mean(tf.square(tf.subtract(model(original), original)))
#     return reconstruction_error
#
#
# def train(loss, model, opt, original):
#     with tf.GradientTape() as tape:
#         gradients = tape.gradient(loss(model, original), model.trainable_variables)
#     gradient_variables = zip(gradients, model.trainable_variables)
#     opt.apply_gradients(gradient_variables)
#
#
# writer = tf.summary.create_file_writer('tmp')
#
# with writer.as_default():
#     with tf.summary.record_if(True):
#         for epoch in range(epochs):
#             for step, batch_features in enumerate(train_data):
#                 train(loss, autoencoder, opt, batch_features)
#                 loss_values = loss(autoencoder, batch_features)
#                 # original = tf.reshape(batch_features, (batch_features.shape[0], 28, 28, 1))
#                 # reconstructed = tf.reshape(autoencoder(tf.constant(batch_features)), (batch_features.shape[0], 28, 28, 1))
#                 tf.summary.scalar('loss', loss_values, step=step)
#                 # tf.summary.image('original', original, max_outputs=10, step=step)
#                 # tf.summary.image('reconstructed', reconstructed, max_outputs=10, step=step)