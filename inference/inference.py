import threading
import datetime
import sys
sys.path.append('../util')
import DataSet
import json
import tensorflow as tf

from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.models import Model, model_from_json
from keras.layers import Dense, GlobalAveragePooling2D


# create the base pre-trained model
def build_model_vgg(nb_classes, num_apps):
    base_model = VGG16(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    shared_output = base_model.output
    top_layers = []
    for i in range(num_apps):
        x = GlobalAveragePooling2D()(shared_output)
        # add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        # add a logistic layer
        x = Dense(nb_classes, activation='softmax')(x)
        top_layers.append(x)

    # this is the model we will train
    model = Model(input=base_model.input, outputs=top_layers)

    print "starting model compile"
    compile(model)
    print "model compile done"
    return model

def build_model(nb_classes, num_apps):
    base_model = InceptionV3(weights="imagenet", include_top=False)

    # add a global spatial average pooling layer
    shared_output = base_model.output
    top_layers = []
    for i in range(num_apps):
        x = GlobalAveragePooling2D()(shared_output)
        # add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        # add a logistic layer
        x = Dense(nb_classes, activation='softmax')(x)
        top_layers.append(x)

    # this is the model we will train
    model = Model(input=base_model.input, outputs=top_layers)

    print "starting model compile"
    compile(model)
    print "model compile done"
    return model

def compile(model):
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])

graph = None # https://github.com/fchollet/keras/issues/2397

def benchmark(dataset_dir, num_apps, batch_size, shared=False):
    data = DataSet.DataSet(dataset_dir, 224)

    '''
    if shared:
        model = build_model(data.nb_classes, num_apps)
        start = datetime.datetime.now()
        t = threading.Thread(target = inference_async, args = [model, data])
        t.start()
        t.join()
        end = datetime.datetime.now()
        '''
    # To warm up some caches or something
    model = build_model(data.nb_classes, num_apps)
    model.predict(data.X_train, batch_size=batch_size)
    
    if shared:
        model = build_model(data.nb_classes, num_apps)
        start = datetime.datetime.now()
        model.predict(data.X_train, batch_size=batch_size)
        end = datetime.datetime.now()
    else:
        models = []
        for x in range(1, num_apps + 1):
            model = build_model(data.nb_classes, 1)
            models.append(model)
            global graph
            graph = tf.get_default_graph()
        threads = []
        for x, model in zip(range(1, num_apps + 1), models):
            start = datetime.datetime.now()
            print "[LOG] Creating thread", x
            t = threading.Thread(target = inference_async, args = [model, data, batch_size])
            print "[LOG] Starting thread", x
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        end = datetime.datetime.now()
    elapsed = end - start
    fps = len(data.X_train) / float(elapsed.seconds)
    print "Shared:", shared, "Num apps: ", num_apps, fps, "fps"
    return fps

def inference_async(model, data, batch_size):
    with graph.as_default():
        model.predict(data.X_train, batch_size=batch_size)

    


