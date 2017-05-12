import sys
sys.path.append('../util')
import DataSet
import threading
import datetime
import json
import tensorflow as tf

from keras import backend as K
from keras.backend.tensorflow_backend import get_session
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.models import Model, model_from_json
from keras.layers import Dense, GlobalAveragePooling2D


# create the base pre-trained model
def build_model(nb_classes):
    base_model = InceptionV3(weights="imagenet", include_top=False)

    # add a global spatial average pooling layer
    shared_output = base_model.output
    x1 = GlobalAveragePooling2D()(shared_output)
    # add a fully-connected layer
    x2 = Dense(1024, activation='relu')(x1)
    # add a logistic layer
    output = Dense(nb_classes, activation='softmax')(x2)

    layer_names = [layer.name for layer in base_model.layers]
    print layer_names
    # this is the model we will train
    model = Model(input=base_model.input, output=x1)
    model._make_predict_function()
    #compile(model)

    return model

def compile(model):
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])

graph = None # https://github.com/fchollet/keras/issues/2397

def run(data, batch_size):

    # To warm up some caches or something
    model = build_model(data.nb_classes)
    output = model.predict(data.X_train, batch_size=batch_size)
    print output

if __name__ == "__main__":
    dataset_dir = sys.argv[1]
    data = DataSet.DataSet(dataset_dir, 224)
    run(data, 64)
