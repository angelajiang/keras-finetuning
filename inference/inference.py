
import json

from keras.applications.inception_v3 import InceptionV3
from keras.models import Model, model_from_json
from keras.layers import Dense, GlobalAveragePooling2D


# create the base pre-trained model
def build_model(nb_classes, weights="imagenet"):
    base_model = InceptionV3(weights=weights, include_top=False)

    # add a global spatial average pooling layer
    shared_output = base_model.output
    x1 = GlobalAveragePooling2D()(shared_output)
    x2 = GlobalAveragePooling2D()(shared_output)
    # let's add a fully-connected layer
    x1 = Dense(1024, activation='relu')(x1)
    x2 = Dense(1024, activation='relu')(x2)
    # and a logistic layer
    predictions1 = Dense(nb_classes, activation='softmax')(x1)
    predictions2 = Dense(nb_classes, activation='softmax')(x2)

    # this is the model we will train
    model = Model(input=base_model.input, outputs=[predictions1, predictions2])

    print "starting model compile"
    compile(model)
    print "model compile done"
    return model


def save(model, tags, prefix):
    model.save_weights(prefix+".h5")
    # serialize model to JSON
    model_json = model.to_json()
    with open(prefix+".json", "w") as json_file:
        json_file.write(model_json)
    with open(prefix+"-labels.json", "w") as json_file:
        json.dump(tags, json_file)


def load(prefix):
    # load json and create model
    with open(prefix+".json") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    # load weights into new model
    model.load_weights(prefix+".h5")
    with open(prefix+"-labels.json") as json_file:
        tags = json.load(json_file)
    return model, tags

def compile(model):
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])
