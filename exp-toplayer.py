import FineTunerFast as ft
import sys
import Train

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

if __name__ == "__main__":
    dataset_name, data_directory, config_file_name, nb_epoch, nb_epoch_batch_size  = sys.argv[1:]
    nb_epoch = int(nb_epoch)
    nb_epoch_batch_size = int(nb_epoch_batch_size)

    optimizers = ["adam", "adamax", "rmsprop"]
    decays_gen = frange(0, .5, .05)
    decays = list(decays_gen)

    optimizers = ["adam"]
    decays = [0.5]
    
    for optimizer in optimizers:
        for decay in decays:
            print "----------------------", optimizer, decay, "----------------------"
            trainer = Train.Train(config_file_name, dataset_name, data_directory, 
                            nb_epoch, nb_epoch_batch_size, optimizer, decay)
            trainer.train()

