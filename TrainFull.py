import DataSet
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

class TrainFull:
    def __init__(self, config_file_path, dataset_name, data_directory, 
                 nb_epoch, nb_epoch_batch_size, optimizer_name,  decay):

        np.random.seed(1337)

        config_parserr = ConfigParser.RawConfigParser()   
        config_parserr.read(config_file_path)

        self.nb_epoch = nb_epoch
        self.nb_epoch_batch_size = nb_epoch_batch_size

        self.n = int(config_parserr.get('full-train-config', 'n'))
        self.batch_size = int(config_parserr.get('full-train-config', 'batch_size'))
        model_dir = str(config_parserr.get('full-train-config', 'model_dir'))
        outfile_dir = str(config_parserr.get('full-train-config', 'outfile_dir'))

        self.dataset = DataSet.DataSet(data_directory, self.n)

        self.weights = str(config_parserr.get('full-train-config', 'weights'))
        if self.weights == "imagenet":
            self.weights = "imagenet"
            weights_name= "imagenet"
        else:
            self.weights = None
            weights_name= "random"

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

        self.model_file_prefix = model_dir + "/" + dataset_name + "-nb" + str(nb_epoch) + ":" + str(nb_epoch_batch_size) + "-" + optimizer_name + "-decay" + str(decay) + "-" + weights_name

        self.outfile = outfile_dir + "/" + dataset_name + "-nb" + str(nb_epoch) + ":" + str(nb_epoch_batch_size) + "-" + optimizer_name + "-decay" + str(decay) + "-" + weights_name
        print self.outfile

    def evaluate(self, model):
        Y_pred = model.predict(self.dataset.X_test, batch_size=self.batch_size)
        y_pred = np.argmax(Y_pred, axis=1)

        accuracy = float(np.sum(self.dataset.y_test==y_pred)) / len(self.dataset.y_test)
        print "accuracy:", accuracy
        
        confusion = np.zeros((self.dataset.nb_classes, self.dataset.nb_classes), dtype=np.int32)
        for (predicted_index, actual_index, image) in zip(y_pred, self.dataset.y_test, self.dataset.X_test):
            confusion[predicted_index, actual_index] += 1
        
        print "rows are predicted classes, columns are actual classes"
        for predicted_index, predicted_tag in enumerate(self.dataset.tags):
            print predicted_tag[:7],
            for actual_index, actual_tag in enumerate(self.dataset.tags):
                print "\t%d" % confusion[predicted_index, actual_index],
        return accuracy

    def train(self):

        f = open(self.outfile, 'w', 0)
        model_file = self.model_file_prefix + ".h5"
        if (not os.path.isfile(model_file)):
            print "Cached model file does not exist"

            model = net.build_model(self.dataset.nb_classes, self.weights)
            # Make all layers trainable
            for layer in model.layers:
               layer.trainable = True
            model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=["accuracy"])

            # train the model on the new data for a few epochs

            num_training_batchs = self.nb_epoch // self.nb_epoch_batch_size

            for i in range(num_training_batchs):
                loss = model.fit(self.dataset.X_train, self.dataset.Y_train,
                                 initial_epoch=0,
                                 batch_size=self.batch_size,
                                 nb_epoch=self.nb_epoch_batch_size,
                                 validation_split=0.1,
                                 shuffle=False)

                acc = self.evaluate(model)
                iters = (i+1) * self.nb_epoch_batch_size
                line = str(iters) + "," + str(acc) + "\n"
                print line
                f.write(line)
                f.flush()

            net.save(model, self.dataset.tags, self.model_file_prefix)

        else:
            print "Model is already cached"



