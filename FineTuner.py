import ConfigParser
import json
import net
import numpy as np
import pprint as pp
import os.path
import sys

from collections import defaultdict

# It's very important to put this import before keras,
# as explained here: Loading tensorflow before scipy.misc seems to cause imread to fail #1541
# https://github.com/tensorflow/tensorflow/issues/1541
import scipy.misc

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras import backend as K
from keras.utils import np_utils

import dataset

class FineTuner:

    def __init__(self, config_file_path, data_directory, model_file_prefix):

        np.random.seed(1337)

        config_parserr = ConfigParser.RawConfigParser()   
        config_parserr.read(config_file_path)

        self.data_directory = data_directory
        self.model_file_prefix = model_file_prefix

        self.n = int(config_parserr.get('finetune-config', 'n'))
        self.batch_size = int(config_parserr.get('finetune-config', 'batch_size'))
        self.nb_epoch = int(config_parserr.get('finetune-config', 'nb_epoch'))
        self.nb_phase_two_epoch = int(config_parserr.get('finetune-config', 'nb_phase_two_epoch'))
        self.num_mega_epochs = int(config_parserr.get('finetune-config', 'num_mega_epochs'))
        self.heavy_augmentation = bool(config_parserr.get('finetune-config', 'heavy_augmentation'))

        self.init_dataset()     # sets self.datagen, self.X_train, self.Y_train, self.X_test, self.Y_test
        self.init_model()       # sets self.model, self.tags

        self.config_parser = config_parserr

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
            print "Cached model file does not exist"

            print "Loading original inception model"
            model = net.build_model(self.nb_classes)
            model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])

            # train the model on the new data for a few epochs

            model.fit_generator(self.datagen.flow(self.X_train, self.Y_train, batch_size=self.batch_size, shuffle=True),
                        samples_per_epoch=self.X_train.shape[0],
                        nb_epoch=self.nb_epoch,
                        validation_data=self.datagen.flow(self.X_test, self.Y_test, batch_size=self.batch_size),
                        nb_val_samples=self.X_test.shape[0]
                        )

            print "evaluating model"
            self.evaluate(model, "000.png")

            print "saving net"
            net.save(model, tags, self.model_file_prefix)

        else:
            print "Load model from cached files"
            model, tags = net.load(self.model_file_prefix)

        self.model = model
        self.tags = tags

    def evaluate(self, model, vis_filename=None):
        print "Predicting..."
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
            print
        if vis_filename is not None:
            bucket_size = 10
            image_size = self.n // 4 # right now that's 56
            vis_image_size = self.nb_classes * image_size * bucket_size
            vis_image = 255 * np.ones((vis_image_size, vis_image_size, 3), dtype='uint8')
            example_counts = defaultdict(int)
            for (predicted_tag, actual_tag, normalized_image) in zip(y_pred, self.y_test, self.X_test):
                example_count = example_counts[(predicted_tag, actual_tag)]
                if example_count >= bucket_size**2:
                    continue
                image = dataset.reverse_preprocess_input(normalized_image)
                image = image.transpose((1, 2, 0))
                image = scipy.misc.imresize(image, (image_size, image_size)).astype(np.uint8)
                tilepos_x = bucket_size * predicted_tag
                tilepos_y = bucket_size * actual_tag
                tilepos_x += example_count % bucket_size
                tilepos_y += example_count // bucket_size
                pos_x, pos_y = tilepos_x * image_size, tilepos_y * image_size
                vis_image[pos_y:pos_y+image_size, pos_x:pos_x+image_size, :] = image
                example_counts[(predicted_tag, actual_tag)] += 1
            vis_image[::image_size * bucket_size, :] = 0
            vis_image[:, ::image_size * bucket_size] = 0
            scipy.misc.imsave(vis_filename, vis_image)
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
        print "compile model"
        self.model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=["accuracy"])

        # we train our model again (this time fine-tuning the top 2 inception blocks
        # alongside the top Dense layers

        print "fine-tuning top 2 inception blocks alongside the top dense layers"

        final_model_file_prefix = self.model_file_prefix + "_final"

        for i in range(1, self.num_mega_epochs + 1):
            print "mega-epoch %d/%d" % (i, self.num_mega_epochs)
            self.model.fit_generator(self.datagen.flow(self.X_train, self.Y_train, batch_size=self.batch_size, shuffle=True),
                    samples_per_epoch=self.X_train.shape[0],
                    nb_epoch=self.nb_phase_two_epoch,
                    validation_data=self.datagen.flow(self.X_test, self.Y_test, batch_size=self.batch_size),
                    nb_val_samples=self.X_test.shape[0]
                    )

            accuracy = self.evaluate(self.model, str(i).zfill(3)+".png")

            net.save(self.model, self.tags, final_model_file_prefix)

        return accuracy

    def print_config(self):
        d = dict(self.config_parser.items("finetune-config"))
        pp.pprint(d)


