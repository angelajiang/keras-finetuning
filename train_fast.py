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
from keras.optimizers import SGD
from keras import backend as K
from keras.utils import np_utils

import dataset
import net

np.random.seed(1337)

n = 224
batch_size = 64

data_directory, model_file_prefix, nb_epoch = sys.argv[1:]
nb_epoch = int(nb_epoch)

print "loading dataset"

X, y, tags = dataset.dataset(data_directory, n)
nb_classes = len(tags)

sample_count = len(y)
train_size = sample_count * 4 // 5
X_train = X[:train_size]
y_train = y[:train_size]
Y_train = np_utils.to_categorical(y_train, nb_classes)
X_test  = X[train_size:]
y_test  = y[train_size:]
Y_test = np_utils.to_categorical(y_test, nb_classes)

X_train = [x.reshape(224, 224, 3) for x in X_train]
X_test = [x.reshape(224, 224, 3) for x in X_test]

def evaluate(model):
    Y_pred = model.predict(X_test, batch_size=batch_size)
    y_pred = np.argmax(Y_pred, axis=1)

    accuracy = float(np.sum(y_test==y_pred)) / len(y_test)
    print "accuracy:", accuracy
    
    confusion = np.zeros((nb_classes, nb_classes), dtype=np.int32)
    for (predicted_index, actual_index, image) in zip(y_pred, y_test, X_test):
        confusion[predicted_index, actual_index] += 1
    
    print "rows are predicted classes, columns are actual classes"
    for predicted_index, predicted_tag in enumerate(tags):
        print predicted_tag[:7],
        for actual_index, actual_tag in enumerate(tags):
            print "\t%d" % confusion[predicted_index, actual_index],
        print
    return accuracy

X_train = np.array(X_train)
X_test = np.array(X_test)

model_file = model_file_prefix + ".h5"
if (not os.path.isfile(model_file)):
    print "Cached model file does not exist"

    model = net.build_model(nb_classes)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])

    # train the model on the new data for a few epochs

    loss = model.fit(X_train, Y_train,
                     initial_epoch=0,
                     batch_size=batch_size,
                     nb_epoch=nb_epoch,
                     validation_data=(X_test, Y_test),
                     shuffle=False)

    evaluate(model)

    net.save(model, tags, model_file_prefix)

else:
    print "Model is already cached"

