import keras

class LossHistory(keras.callbacks.Callback):
    def __init__(self, history_file, num_train):
        self.num_epochs = 0
        self.num_train = num_train
        self.f = open(history_file, 'a+', 0)

    def on_train_begin(self, logs={}):
        self.num_epochs = 0

    def on_epoch_end(self, batch, logs={}):
        self.num_epochs += 1
        val_loss = logs.get('val_loss')
        val_loss = str.format("{0:.4f}", val_loss)
        val_acc = logs.get('val_acc')
        val_acc = str.format("{0:.4f}", val_acc)
        line = str(self.num_train) + "," + str(self.num_epochs) + "," + str(val_loss) + "," + str(val_acc) + "\n"
        self.f.write(line)
        self.f.flush()
