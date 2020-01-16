from tf.keras.models import Sequential, Model
from tf.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

# Defining a model using Keras
# Sequential API
model = Sequential([
	Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
	MaxPooling2D(pool_size=(2, 2)),
	Flatten(),
	Dense(10, activation='softmax')
])

# functional API
inputs = Input(shape=(32, 32, 3))
x = Conv2D(32, (3, 3), activation='relu')(inputs)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dense(10, activation='softmax')(x)
model = Model(inputs=inputs, outputs=x)

# TensorFlow's official Keras implementation of ResNet
input_shape = (32, 32, 3)
img_input = Input(shape=input_shape)
model = resnet_cifar_model.resnet56(img_input, classes=10)

# Data pipeline
# Load CIFAR-10 from storage to memory
(x, y), (x_test, y_test) = keras.datasets.cifar10.load_data()
print(type(x), type(y))
print(x.shape, y.shape)

# Instantiate the Dataset class
train_dataset = tf.data.Dataset.from_tensor_slices((x, y))

# Data augmentation
def augmentation(x, y):
	x = tf.image.resize_with_crop_or_pad(
		x, HEIGHT + 8, WIDTH + 8)
	X = tf.image.random_crop(x, [HEIGHT, WIDTH, NUM_CHANNELS])
	x = tf.image.random_flip_left_right(x)
	return x, y

train_dataset = train_dataset.map(augmentation)

# Shuffling
train_dataset = (train_dataset
				.map(augmentation)
				.shuffle(buffer_size=50000))

# Normalization
def normalize(x, y):
	x = tf.image.per_image_standardization(x)
	return x, y

train_dataset = (train_dataset
				.map(augmentation)
				.shuffle(buffer_size=50000)
				.map(normalize))

# Batching
train_dataset = (train_dataset
				.map(augmentation)
				.map(normalize)
				.shuffle(buffer_size=50000)
				.batch(128, drop_remainder=True)