from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from PIL import Image
from random import randrange
import pdb


# Erreur dans la localisation des images empêchants leur labelling


class PathError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class DataGenerator:
    def __init__(self, images_path, images_size, batch, ratio_test_train=.2, number_calls=None, zoom_range=False, image_flip=False,
                 translation_x_range=False, translation_y_range=False, brightness_value_range=False):
        self.data_set_train = []
        self.data_set_valid = []
        self.lenth_data_train = None
        self.lenth_data_valid = None
        self.labels = []
        self.images_path = images_path
        self.image_size = images_size
        self.batch_size = batch
        self.ratio_test_train = ratio_test_train
        self.number_calls = number_calls
        self.zoom_range = zoom_range
        self.image_flip = image_flip
        self.translation_x_range = translation_x_range
        self.translation_y_range = translation_y_range
        self.brightness_value_range = brightness_value_range

        self.make_labels()
        self.fill_data_set()

    def make_labels(self):
        for filename in self.images_path:
            if 'cat' in filename:
                self.labels.append([1, 0])
            elif 'dog' in filename:
                self.labels.append([0, 1])
            else:
                raise PathError("l'image à l'adresse suivante %s a été mis dans un mauvais dossier" % filename)

    def fill_data_set(self):
        x_train, x_valid, y_train, y_valid = train_test_split(self.images_path, self.labels,
                                                              test_size=self.ratio_test_train)

        x_train = list(map(self.open_image, x_train))
        x_valid = list(map(self.open_image, x_valid))
        self.lenth_data_train = tf.convert_to_tensor(len(x_train), dtype=tf.int64)
        self.lenth_data_valid = tf.convert_to_tensor(len(x_valid), dtype=tf.int64)

        with tf.device('CPU:0'):
            self.data_set_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(self.batch_size)
            self.data_set_valid = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(self.batch_size)

    def open_image(self, path):
        with Image.open(path) as img:
            img = img.resize(self.image_size)
            return np.array(img, np.float32) / 255

    def get_train_data(self):
        self.data_set_train.map(self.process_images, num_parallel_calls=self.number_calls)
        return self.data_set_train.shuffle(buffer_size=self.lenth_data_train, seed=42)
        # return self.data_set_train

    def get_valid_data(self):
        self.data_set_valid.map(self.process_images, num_parallel_calls=self.number_calls)
        return self.data_set_valid.shuffle(buffer_size=self.lenth_data_train, seed=42)
        # return self.data_set_valid

    def it_train_data(self):
        return IteratorDataGenerator(self.data_set_train, self)

    def it_valid_data(self):
        return IteratorDataGenerator(self.data_set_valid, self)

    def process_images(self, image, labels):
        #TODO faire fonctionner la fonction

        if self.zoom_range is not False:
            if type(self.zoom_range) is list:
                zoom_value = float(randrange(self.zoom_range[0], self.zoom_range[1]))
                # image = zoom(image, zoom_value)
            else:
                Exception('Bad value input for zoom_value, need a list but received a ' + str(type(self.zoom_value)))

        if self.translation_x_range is not False:
            if type(self.translation_x_range) is list:
                translation_x_value = randrange(self.translation_x_range[0], self.translation_x_range[1])
            else:
                Exception('Bad value input for translation_x_range, need a list but received a ' + str(
                    type(self.translation_x_range)))

        if self.translation_y_range is not False:
            if type(self.translation_y_range) is list:
                translation_y_value = randrange(self.translation_y_range[0], self.translation_y_range[1])
            else:
                Exception('Bad value input for translation_y_range, need a list but received a ' + str(
                    type(self.translation_y_range)))

        if self.brightness_value_range is not False:
            if type(self.brightness_value_range) is list:
                brightness_value_value = randrange(self.brightness_value_range[0], self.brightness_value_range[1])
                tf.image.adjust_brightness(brightness_value_value)
            else:
                Exception('Bad value input for brightness_value_range, need a list but received a ' + str(
                    type(self.brightness_value_range)))

        if self.image_flip is not False:
            if self.image_flip is True:
                image = tf.image.random_flip_left_right(image, seed=42)
            else:
                Exception('Bad value input for image_flip, need a list but received a ' + str(type(self.image_flip)))

        return tf.convert_to_tensor(image, dtype=tf.float32), labels


class IteratorDataGenerator:
    def __init__(self, data_set, parent):
        self.data_set = data_set
        self.parent = parent
        self.advancement = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.advancement == len(self.data_set):
            raise StopIteration

        i = 0
        data_set_temp = []

        for set_image, set_valid in self.data_set.skip(i):
            pdb.set_trace()

        self.advancement += 1

        return tuple(map(list, zip(*data_set_temp)))