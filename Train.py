import ConfigParser
import json
import numpy as np
import os.path
import sys

from collections import defaultdict

# It's very important to put this import before keras,
# as explained here: Loading tensorflow before scipy.misc seems to cause imread to fail #1541
# https://github.com/tensorflow/tensorflow/issues/1541
import scipy.misc

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam, Nadam, RMSprop, Adamax
from keras import backend as K
from keras.utils import np_utils

import dataset
import net

np.random.seed(1337)

class Train:
    def __init__(self, config_file_path, dataset_name, data_directory, 
                 nb_epoch, nb_epoch_batch_size, optimizer_name,  decay):

        np.random.seed(1337)

        config_parserr = ConfigParser.RawConfigParser()   
        config_parserr.read(config_file_path)

        self.data_directory = data_directory
        self.nb_epoch = nb_epoch
        self.nb_epoch_batch_size = nb_epoch_batch_size

        self.n = int(config_parserr.get('top-layer-config', 'n'))
        self.batch_size = int(config_parserr.get('top-layer-config', 'batch_size'))
        self.heavy_augmentation  = bool(config_parserr.get('top-layer-config', 'heavy_augmentation'))
        model_dir = str(config_parserr.get('top-layer-config', 'model_dir'))
        outfile_dir = str(config_parserr.get('top-layer-config', 'outfile_dir'))

        if optimizer_name == "adam":
            self.optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
        elif optimizer_name == "adamax":
            self.optimizer = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
        elif optimizer_name == "nadam":
            self.optimizer = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=decay)
        elif optimizer_name == "rmsprop":
            self.optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=decay)
        elif optimizer_name == "sgd":
            self.optimizer = SGD(lr=0.01, momentum=0.0, decay=decay, nesterov=False)
        else:
            print "[ERROR] Didn't recognize optimizer", optimizer_name
	    sys.exit(-1)

        self.model_file_prefix = model_dir + "/" + dataset_name + "-nb" + str(nb_epoch) + ":" + str(nb_epoch_batch_size) + "-" + optimizer_name + "-decay" + str(decay)
        self.outfile = outfile_dir + "/" + dataset_name + "-nb" + str(nb_epoch) + ":" + str(nb_epoch_batch_size) + "-" + optimizer_name + "-decay" + str(decay)
        print self.outfile

        self.init_dataset()

    def init_dataset(self):

        X, y, tags = dataset.dataset(self.data_directory, self.n)
        self.nb_classes = len(tags)

        sample_count = len(y)
        train_size = sample_count * 4 // 5
        X_train = X[:train_size]
        y_train = y[:train_size]
        Y_train = np_utils.to_categorical(y_train, self.nb_classes)
        X_test  = X[train_size:]
        y_test  = y[train_size:]
        Y_test = np_utils.to_categorical(y_test, self.nb_classes)

        X_train = [x.reshape(224, 224, 3) for x in X_train]
        X_test = [x.reshape(224, 224, 3) for x in X_test]

        if self.heavy_augmentation:
            datagen = ImageDataGenerator(
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                rotation_range=45,
                width_shift_range=0.25,
                height_shift_range=0.25,
                horizontal_flip=True,
                vertical_flip=False,
                zoom_range=0.5,
                channel_shift_range=0.5,
                fill_mode='nearest')
        else:
            datagen = ImageDataGenerator(
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                rotation_range=0,
                width_shift_range=0.125,
                height_shift_range=0.125,
                horizontal_flip=True,
                vertical_flip=False,
                fill_mode='nearest')

        self.X_train = np.array(X_train)
        self.X_test = np.array(X_test)
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.y_train = y_train
        self.y_test = y_test
        self.tags = tags

    def evaluate(self, model):
        Y_pred = model.predict(self.X_test, batch_size=self.batch_size)
        y_pred = np.argmax(Y_pred, axis=1)

        accuracy = float(np.sum(self.y_test==y_pred)) / len(self.y_test)
        print "accuracy:", accuracy
        
        confusion = np.zeros((self.nb_classes, self.nb_classes), dtype=np.int32)
        for (predicted_index, actual_index, image) in zip(y_pred, self.y_test, self.X_test):
            confusion[predicted_index, actual_index] += 1
        
        print "rows are predicted classes, columns are actual classes"
        for predicted_index, predicted_tag in enumerate(self.tags):
            print predicted_tag[:7],
            for actual_index, actual_tag in enumerate(self.tags):
                print "\t%d" % confusion[predicted_index, actual_index],
        return accuracy

    def train(self):

        f = open(self.outfile, 'w', 0)
        model_file = self.model_file_prefix + ".h5"
        if (not os.path.isfile(model_file)):
            print "Cached model file does not exist"

            model = net.build_model(self.nb_classes)
            model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=["accuracy"])

            # train the model on the new data for a few epochs

            num_training_batchs = self.nb_epoch // self.nb_epoch_batch_size

            for i in range(num_training_batchs):
                loss = model.fit(self.X_train, self.Y_train,
                                 initial_epoch=0,
                                 batch_size=self.batch_size,
                                 nb_epoch=self.nb_epoch_batch_size,
                                 validation_data=(self.X_test, self.Y_test),
                                 shuffle=False)

                acc = self.evaluate(model)
                iters = i * self.nb_epoch_batch_size
                line = str(iters) + "," + str(acc) + "\n"
                print line
                f.write(line)
                f.flush()

            net.save(model, self.tags, self.model_file_prefix)

        else:
            print "Model is already cached"



