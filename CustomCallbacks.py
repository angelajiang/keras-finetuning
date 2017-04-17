import keras

class LossHistory(keras.callbacks.Callback):
    def __init__(self, history_file):
        self.num_epochs = 0
        self.f = open(history_file, 'w', 0)

    def on_train_begin(self, logs={}):
        self.num_epochs = 0

    def on_epoch_end(self, batch, logs={}):
        self.num_epochs += 1
        acc = logs.get('acc')
        acc = str.format("{0:.4f}", acc)
        val_acc = logs.get('val_acc')
        val_acc = str.format("{0:.4f}", val_acc)
        line = str(self.num_epochs) + "," + str(acc) + "," + str(val_acc) + "\n"
        self.f.write(line)
        self.f.flush()
