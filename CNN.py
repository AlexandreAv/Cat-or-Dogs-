#!/usr/bin/env python
# coding: utf-8

# # Mise à jour vers Tensorflow 2.0

# In[1]:


import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from glob import glob
from data import DataGenerator

tf.debugging.set_log_device_placement(True)
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# ## Hyperparamètres

# In[2]:


LEARNING_RATE = 0.0001
EPOCH = 100
BATCH_SIZE = 250
NUMBER_PREDICTIONS = 2
TEST_SIZE = 0.2

# ## Charger les images et on exécute le préprocessing des images, 1er étape

# In[3]:

IMAGE_WIDTH = 200
IMAGE_HEIGHT = 200

path = glob('C:/dataset/train/*.jpg')
# path = ('./dataset/train/cat.0.jpg', './dataset/train/cat.1.jpg', './dataset/train/cat.dog.2.jpg', './dataset/train/cat.3.jpg',
#         './dataset/train/dog.0.jpg', './dataset/train/dog.1.jpg', './dataset/train/dog.dog.2.jpg', './dataset/train/dog.3.jpg')

dataset = DataGenerator(path, (IMAGE_WIDTH, IMAGE_HEIGHT), BATCH_SIZE, TEST_SIZE, number_calls=4, image_flip=True, brightness_value_range=(0, 1))


# ## On affiche quelques images de la dataset après la 1er étape du proprocessing

# In[4]:


# batch = dataset.get_valid_data()

# for i in range(3):
#     imread(batch[i][0][0])
#     plt.show()


# ## On crée notre modèle à convolution 

# In[5]:


# Création du model avec le Subclassing API
class ConvModel(tf.keras.Model):
    def __init__(self):
        super(ConvModel, self).__init__()
        self.conv1 = Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3))
        self.pool1 = MaxPool2D((2, 2))
        self.dropout1 = Dropout(0.2)
        self.conv2 = Conv2D(64, (3, 3), activation='relu')
        self.pool2 = MaxPool2D((3, 3))
        self.dropout2 = Dropout(0.2)
        self.conv3 = Conv2D(64, (3, 3), activation='relu')
        self.pool3 = MaxPool2D((3, 3))
        # self.dropout3 = Dropout(0.3)
        self.flatten = Flatten(name='flatten')
        self.dens1 = Dense(128, activation='relu', name='dens1')
        self.out = Dense(NUMBER_PREDICTIONS, activation='softmax', name='output')

    def call(self, image):
        x = self.conv1(image)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        # x = self.dropout3(x)
        x = self.flatten(x)
        x = self.dens1(x)

        return self.out(x)


model = ConvModel()

# ## On initialise la fonction de perte et l'optimizer

# In[6]:


loss_object = CategoricalCrossentropy()
optimizer = Adam(lr=LEARNING_RATE)

# ## On initialise des metrics, soit des moyennes

# In[7]:


metrics_train_loss = tf.metrics.CategoricalCrossentropy()
metrics_train_accuracy = tf.metrics.Accuracy()

metrics_valid_loss = tf.metrics.CategoricalCrossentropy()
metrics_valid_accuracy = tf.metrics.Accuracy()


# ## On crée la fonction de teste

# In[8]:


@tf.function
def train(images, targets):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(targets, predictions)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    metrics_train_loss(targets, predictions)
    metrics_train_accuracy(tf.math.argmax(targets, axis=1), tf.math.argmax(predictions, axis=1))


# ## On crée la fonction de validation  

# In[9]:


@tf.function
def valid_step(images, targets):
    predictions = model(images)
    loss = loss_object(targets, predictions)

    metrics_valid_loss(targets, predictions)
    metrics_valid_accuracy(tf.math.argmax(targets, axis=1), tf.math.argmax(predictions, axis=1))


for epoch in range(EPOCH):
    train_set = dataset.get_train_data()
    valid_set = dataset.get_valid_data()

    for x_train, y_train in train_set:
        train(tf.convert_to_tensor(x_train, dtype=tf.float32), tf.convert_to_tensor(y_train))

    for x_valid, y_valid in valid_set:
        valid_step(tf.convert_to_tensor(x_valid, dtype=tf.float32), tf.convert_to_tensor(y_valid))

    template = "\n Epoch {}/{}, Loss: {}, Accuracy: {}%"
    print(template.format(epoch + 1, EPOCH, metrics_train_loss.result(), metrics_train_accuracy.result() * 100))

    template = "Valid Loss: {}, Valid Accuracy: {}%"
    print(template.format(metrics_valid_loss.result(), metrics_valid_accuracy.result() * 100))

    if metrics_valid_accuracy.result() > 0.80:
        model.save_weights('save/model.tf')

    metrics_train_loss.reset_states()
    metrics_train_accuracy.reset_states()
    metrics_valid_loss.reset_states()
    metrics_valid_accuracy.reset_states()
