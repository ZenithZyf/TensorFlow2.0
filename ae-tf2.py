import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow.keras.backend as kb

import DOT_preprocess

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True


# def custom_loss(meas_true, meas_pred):
    # tf.reduce_mean(tf.square(tf.subtract(model(original), original)))
    # print(meas_true.shape)
    # print(meas_pred.shape)
    # custom_loss = kb.mean(kb.sqrt(meas_pred-meas_true))
    # return custom_loss

def custom_loss(meas_pred, meas_true):
    custom_loss1 = tf.reduce_mean(tf.square(tf.subtract(meas_pred, meas_true)))
    return custom_loss1

ref_a, meas_simu, batch_ys_a, depth_a = DOT_preprocess.dataprocess()
ref_t, meas_phan, test_image = DOT_preprocess.test_data()

TRAINING_EPOCH = 250
LEARNING_RATE = 1e-7
BATCH_SIZE = 128

train_data = tf.data.Dataset.from_tensor_slices(tf.cast(meas_simu, tf.float64)).shuffle(21654).batch(54)
test_data = tf.data.Dataset.from_tensor_slices(tf.cast(meas_phan, tf.float64))

model = Sequential([
    layers.Dense(64, input_shape=(252,), activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(768, activation='relu', name='mua'),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(252)
])

optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE)

# Iterate over epochs.
for epoch in range(5):
    print(f'Epoch {epoch+1}')

  # Iterate over the batches of the dataset.
    for step, x_batch in enumerate(train_data):
        with tf.GradientTape() as tape:
          recon = model(x_batch)
          loss = mse_loss(x_batch, recon)

        grads = tape.gradient(loss, autoencoder.trainable_variables)
        optimizer.apply_gradients(zip(grads, autoencoder.trainable_variables))

        loss_metric(loss)

        if step % 100 == 0:
          print(f'Step {step}: mean loss = {loss_metric.result()}')



# model.compile(
#           loss=custom_loss,
#           optimizer=optimizer)
#
# model.fit(tf.cast(meas_simu, tf.float64), tf.cast(meas_simu, tf.float64),
#           batch_size=BATCH_SIZE, epochs=TRAINING_EPOCH, shuffle=True)
#
# earlyPredictor = tf.keras.Model(model.inputs, model.get_layer('mua').output)
#
# result = earlyPredictor.predict(meas_phan).reshape((12,256,3))

print("Finished!")

