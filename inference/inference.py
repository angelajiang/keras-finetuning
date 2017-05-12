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
    model._make_predict_function()
    compile(model)

    return model

def build_model(nb_classes, num_apps):
    base_model = InceptionV3(weights="imagenet", include_top=False)

    # add a global spatial average pooling layer
    shared_output = base_model.output
    #x = GlobalAveragePooling2D()(shared_output)
    top_layers = []
    for i in range(num_apps):
        x = GlobalAveragePooling2D()(shared_output)
        # add a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        # add a logistic layer
        x = Dense(nb_classes, activation='softmax')(x)
        top_layers.append(x)

    if len(top_layers) == 0:
        top_layers.append(shared_output)

    # this is the model we will train
    model = Model(input=base_model.input, outputs=top_layers)
    model._make_predict_function()
    #compile(model)

    return model

def build_model_by_layer(nb_classes, num_layers):
    base_model = InceptionV3(weights="imagenet", include_top=False)
    shared_output = base_model.output
    x = GlobalAveragePooling2D()(shared_output)
    # add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # add a logistic layer
    x = Dense(nb_classes, activation='softmax')(x)
    full_model = Model(input=base_model.input, output=x)
    output = full_model.layers[num_layers].output
    model = Model(input=full_model.input, output=output)
    return model

def compile(model):
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])

graph = None # https://github.com/fchollet/keras/issues/2397

def benchmark_runtime(data, batch_size, trials):

    # To warm up some caches or something
    runtimes_by_layer = {}
    for num_layers in range(1, 314, 10):
        model = build_model_by_layer(data.nb_classes, num_layers)
        runtimes = []
        for i in range(trials):
            start = datetime.datetime.now()
            model.predict(data.X_train, batch_size=batch_size)
            end = datetime.datetime.now()
            elapsed = end - start
            runtimes.append(elapsed.microseconds)
        average_runtime = sum(runtimes) / float(len(runtimes))
        fps = len(data.X_train) / float(average_runtime) * 1000000
        print "Num layers:", num_layers, "Avg runtime:", average_runtime, "Speed:", fps, "fps"
        runtimes_by_layer[num_layers] = average_runtime
    return runtimes_by_layer

def benchmark(data, num_apps, batch_size, shared=False):

    # To warm up some caches or something
    model = build_model(data.nb_classes, num_apps)
    model.predict(data.X_train, batch_size=batch_size)

    if shared:
        model = build_model(data.nb_classes, num_apps)
        graph = tf.get_default_graph()
        t = threading.Thread(target = inference_async, args = [model, data, batch_size, graph])
        start = datetime.datetime.now()
        t.start()
        t.join()
        end = datetime.datetime.now()
    else:
        threads = []
        model = build_model(data.nb_classes, 1)
        graph = tf.get_default_graph()
        get_session().run(tf.global_variables_initializer())
        get_session().run(tf.local_variables_initializer())
        for x in range(1, num_apps + 1):
            t = threading.Thread(target = inference_async, args = [model, data, batch_size, graph])
            threads.append(t)
        start = datetime.datetime.now()
        for x, t in zip(range(1, num_apps + 1), threads):
            print "[LOG] Starting thread", x
            t.start()
        for t in threads:
            t.join()
        end = datetime.datetime.now()
    elapsed = end - start
    fps = len(data.X_train) / float(elapsed.seconds)
    print "Shared:", shared, "Num apps:", num_apps, "Elapsed:", elapsed, "Speed:", fps, "fps"
    return fps

def inference_async(model, data, batch_size, graph):
    #tf_session = K.get_session()
    #with tf_session.as_default():
    #g = tf.Graph()
    #print "[INFO 2]", g, graph
    with graph.as_default():
        model.predict(data.X_train, batch_size=batch_size)

