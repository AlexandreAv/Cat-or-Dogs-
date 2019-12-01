from sklearn.model_selection import train_test_split
from scipy.ndimage import rotate, zoom
import numpy as np
from PIL import Image
from glob import glob
from copy import deepcopy
from random import randrange
import pdb


# Erreur dans la localisation des images empêchants leur labelling


class PathError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class DataGenerator:
    def __init__(self, images_path, images_size, batch, ratio_test_train=.2, zoom_range=False, image_flip=False,
                 translation_x_range=False, translation_y_range=False, brightness_value_range=False):
        self.data_set_train = []
        self.data_set_valid = []
        self.labels = []
        self.images_path = images_path
        self.image_size = images_size
        self.batch_size = batch
        self.ratio_test_train = ratio_test_train
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
        self.data_set_train = list(map(lambda x: list(x), list(zip(x_train, y_train))))
        self.data_set_valid = list(map(lambda x: list(x), list(zip(x_valid, y_valid))))
        # pdb.set_trace()
        self.data_set_train = [self.data_set_train[x:x + self.batch_size] for x in
                               range(0, len(self.data_set_train), self.batch_size)]
        self.data_set_valid = [self.data_set_valid[x:x + self.batch_size] for x in
                               range(0, len(self.data_set_valid), self.batch_size)]

    def get_train_data(self):
        return self.data_set_train

    def get_valid_data(self):
        return self.data_set_valid

    def it_train_data(self):
        return IteratorDataGenerator(self.data_set_train, self)

    def it_valid_data(self):
        return IteratorDataGenerator(self.data_set_valid, self)

    def process_images(self, image):
        image = image.resize(self.image_size)

        if self.zoom_range is not False:
            if type(self.zoom_range) is list:
                zoom_value = float(randrange(self.zoom_range[0], self.zoom_range[1]))
                image = zoom(image, zoom_value)
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
            else:
                Exception('Bad value input for brightness_value_range, need a list but received a ' + str(
                    type(self.brightness_value_range)))

        if self.image_flip is not False:
            if self.image_flip is True:
                image = np.flip(image)
            else:
                Exception('Bad value input for image_flip, need a list but received a ' + str(type(self.image_flip)))

        return np.array(image)


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

        for data in self.data_set[self.advancement]:
            with Image.open(data[0]) as img:
                img = self.parent.process_images(img)
                data_set_temp.append([img, data[1]])
                i += 1

        self.advancement += 1

        return tuple(map(list, zip(*data_set_temp)))