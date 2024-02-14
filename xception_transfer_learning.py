# Import
import pathlib
import os
import datetime

import PIL
import PIL.Image
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Load the TensorBoard notebook extension
%load_ext tensorboard

# Print TensorFlow version
print(tf.__version__)


# Prepare section
# ---------------

# set data directory
data_dir = './dataset/256x'
data_dir = pathlib.Path(data_dir)

# check image count (all sub directories)
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

# set parameters
batch_size = 32
img_height = 256
img_width = 256


# Load data section
# -----------------

# load training set
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# load validation set
validation_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# Check & resize data section
# ---------------------------

# check training set class names
class_names = train_ds.class_names
print(class_names)

# check a few images from training set
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

# set target size of the image (to feed model)
size = (150, 150)

# resize training and validation set
train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, size), y))
validation_ds = validation_ds.map(lambda x, y: (tf.image.resize(x, size), y))


# set batch size & cache
# ----------------------

# set batch size
#batch_size = 32

# set cache
#train_ds = train_ds.cache().batch(batch_size).prefetch(buffer_size=10)
#validation_ds = validation_ds.cache().batch(batch_size).prefetch(buffer_size=10)
#test_ds = test_ds.cache().batch(batch_size).prefetch(buffer_size=10)


# Data augmentation
# -----------------

#datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#    featurewise_center=True,
 #   featurewise_std_normalization=True,
 #   rotation_range=20,
 #   width_shift_range=0.2,
 #   height_shift_range=0.2,
 #   horizontal_flip=True,
 #   validation_split=0.2)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
#datagen.fit(train_ds)

# visu
#import numpy as np

#for images, labels in train_ds.take(1):
#    plt.figure(figsize=(10, 10))
#    first_image = images[0]
#    for i in range(9):
#        ax = plt.subplot(3, 3, i + 1)
#        augmented_image = data_augmentation(
#            tf.expand_dims(first_image, 0), training=True
#        )
#        plt.imshow(augmented_image[0].numpy().astype("int32"))
#        plt.title(int(labels[0]))
#        plt.axis("off")


# Define model
# ------------

# Define the base model for transfer learning
base_model = tf.keras.applications.Xception(
    weights="imagenet",  # Load weights pre-trained on ImageNet.
    input_shape=(150, 150, 3),
    include_top=False,
)  # Do not include the ImageNet classifier at the top.

# Freeze the base_model
base_model.trainable = False

# Create new model on top
inputs = tf.keras.Input(shape=(150, 150, 3))
x = inputs
#x = data_augmentation(inputs)  # Apply random data augmentation -- update above LINE

# Rescale input (0, 255) to a range of (-1., +1.)
# outputs: `(inputs * scale) + offset`
scale_layer = tf.keras.layers.experimental.preprocessing.Rescaling(scale=1 / 127.5, offset=-1)
x = scale_layer(x)

# The base model contains batchnorm layers. We want to keep them in inference mode
# when we unfreeze the base model for fine-tuning, so we make sure that the
# base_model is running in inference mode here.
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)  # Regularize with dropout
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs, outputs)

# display model summary
model.summary()


# Compile & train model
# ---------------------

# define compile options
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.BinaryAccuracy()])

# set number of epochs
epochs = 20

# set callback to save logs (TensorBoard)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# train model
history = model.fit(
    train_ds,
    epochs = epochs,
    validation_data = validation_ds,
    callbacks = [tensorboard_callback])


# Check accuracy & loss training history
# --------------------------------------

# summarize history for accuracy
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# Final round of model training
# -----------------------------

# Unfreeze the base_model. Note that it keeps running in inference mode
# since we passed `training=False` when calling it. This means that
# the batchnorm layers will not update their batch statistics.
# This prevents the batchnorm layers from undoing all the training
# we've done so far.
base_model.trainable = True
model.summary()

model.compile(
    optimizer = tf.keras.optimizers.Adam(1e-5),  # Low learning rate
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics = [tf.keras.metrics.BinaryAccuracy()],
)

epochs = 10
model.fit(train_ds, epochs=epochs, validation_data=validation_ds)


# BEGIN of test section
# ---------------------

# helper: function to make prediction
def makePrediction(img_url):
  test_ds = tf.keras.utils.load_img(img_url, target_size=(150, 150))

  input_arr = tf.keras.preprocessing.image.img_to_array(test_ds)
  input_arr = np.array([input_arr])  # Convert single image to a batch.

  predictions = model.predict(input_arr)

  # Apply a sigmoid since our model returns logits
  predictions = tf.nn.sigmoid(predictions)
  predictions = tf.where(predictions < 0.5, 0, 1)

  return predictions.numpy()[0][0]

# define test directory
test_dir = './dataset/256x_test'
test_dir = pathlib.Path(test_dir)

# check image count (same directory)
image_list = list(test_dir.glob('*.jpg'))
image_count = len(image_list)
print(image_count)

# define list
predictions = []

# loop to make predictions
for img in image_list:
  print('Image:', img)

  result = makePrediction(img)
  predictions.append(result)

# helper function
def img_load(img):
  img = os.path.join(test_dir, img)
  img = PIL.Image.open(img)
  img = img.resize((150, 150))
  img = np.asarray(img)
  return img

images = os.listdir(test_dir)
img_arr = []

for image in images:
  img_arr.append(img_load(image))

# define labels
labels = ['espresso', 'longblack']

# check test images & prediction label
plt.figure(figsize=(10, 10))
for i in range(len(img_arr)):
    ax = plt.subplot(3, 4, i + 1)
    plt.imshow(img_arr[i])
    plt.title(labels[predictions[i]])
    plt.axis("off")

%tensorboard --logdir logs/fit