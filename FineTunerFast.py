import ConfigParser
import json
import net
import numpy as np
import pprint as pp
import os.path
import sys
import time

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

class FineTunerFast:

    def __init__(self, config_file_path, data_directory, model_file_prefix):

        np.random.seed(1337)

        config_parserr = ConfigParser.RawConfigParser()   
        config_parserr.read(config_file_path)

        self.data_directory = data_directory
        self.model_file_prefix = model_file_prefix

        self.n = int(config_parserr.get('finetune-config', 'n'))
        self.batch_size = int(config_parserr.get('finetune-config', 'batch_size'))
        self.nb_epoch = int(config_parserr.get('finetune-config', 'nb_epoch'))
        self.num_mega_epochs = int(config_parserr.get('finetune-config', 'num_mega_epochs'))
        self.data_augmentation = bool(int(config_parserr.get('finetune-config', 'data_augmentation')))
        self.heavy_augmentation = bool(int(config_parserr.get('finetune-config', 'heavy_augmentation')))
        optimizer_name = str(config_parserr.get('finetune-config', 'optimizer'))
        decay = float(config_parserr.get('finetune-config', 'decay'))
        lr = float(config_parserr.get('finetune-config', 'learning-rate'))                          # Should be low for finetuning
        
        if optimizer_name == "adam":
            self.optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
        elif optimizer_name == "adamax":
            self.optimizer = Adamax(lr=l4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
        elif optimizer_name == "nadam":
            self.optimizer = Nadam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=decay)
        elif optimizer_name == "rmsprop":
            self.optimizer = RMSprop(lr=lr, rho=0.9, epsilon=1e-08, decay=decay)
        elif optimizer_name == "sgd":
            self.optimizer = SGD(lr=lr, momentum=0.0, decay=decay, nesterov=False)
        else:
            print "[ERROR] Didn't recognize optimizer", optimizer_name
	    sys.exit(-1)

        self.init_dataset()     # sets self.datagen, self.X_train, self.Y_train, self.X_test, self.Y_test
        self.init_model()       # sets self.model, self.tags

        self.config_parser = config_parserr

    def init_dataset(self):

        X, y, tags = dataset.dataset(self.data_directory, self.n)
        self.tags = tags
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

        datagen.fit(X_train)

        self.X_train = np.array(X_train)
        self.X_test = np.array(X_test)
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.y_test = y_test
        self.datagen = datagen

    def init_model(self):
        model_file = self.model_file_prefix + ".h5"
        if (not os.path.isfile(model_file)):
            print "[WARNING] Generating new model"
            model = net.build_model(self.nb_classes)

        else:
            print "Load model from cached files"
            model, tags = net.load(self.model_file_prefix)

        model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=["accuracy"])
        net.save(model, self.tags, self.model_file_prefix)
        self.model = model

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

    def finetune(self, num_train):

        # at this point, the top layers are well trained and we can start fine-tuning
        # convolutional layers from inception V3. We will freeze the bottom N layers
        # and train the remaining top layers.
        
        # we chose to train the top 2 inception blocks, i.e. we will freeze
        # the first 172 layers and unfreeze the rest:
        num_layers = len(self.model.layers)
        num_frozen = num_layers - num_train
        if (num_frozen < 0):
            print "[ERROR]: num_train > num_layers"
            return -1

        print "Num_layers:", num_layers, " num frozen:", num_frozen
        for layer in self.model.layers[:num_frozen]:
           layer.trainable = False
        for layer in self.model.layers[num_frozen:]:
           layer.trainable = True

        # we need to recompile the model for these modifications to take effect
        # we use SGD with a low learning rate
        self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=["accuracy"])

        # we train our model again (this time fine-tuning the top 2 inception blocks
        # alongside the top Dense layers

        if self.data_augmentation:
            for i in range(1, self.num_mega_epochs + 1):
                print "mega-epoch %d/%d" % (i, self.num_mega_epochs)
                self.model.fit_generator(self.datagen.flow(self.X_train, self.Y_train, batch_size=self.batch_size, shuffle=False),
                        samples_per_epoch=self.X_train.shape[1],
                        nb_epoch=self.nb_epoch,
                        validation_data=self.datagen.flow(self.X_test, self.Y_test, batch_size=self.batch_size),
                        nb_val_samples=self.X_test.shape[0]
                        )
        else:
            for i in range(1, self.num_mega_epochs + 1):
                print "mega-epoch %d/%d" % (i, self.num_mega_epochs)

                 #   # train the model on the new data for a few epochs
                loss = self.model.fit(self.X_train, self.Y_train,
                                      batch_size=self.batch_size,
                                      nb_epoch=self.nb_epoch,
                                      shuffle=False)

        accuracy = self.evaluate(self.model)

        final_model_file_prefix = self.model_file_prefix + "_final"
        net.save(self.model, self.tags, final_model_file_prefix)
        print "[finetune] Saving model to", final_model_file_prefix

        print "[finetune] accuracy:" , accuracy
        return accuracy

    def print_config(self):
        d = dict(self.config_parser.items("finetune-config"))
        pp.pprint(d)


