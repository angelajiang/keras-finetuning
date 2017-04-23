import sys
sys.path.append('../util')
import TrainFull

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

if __name__ == "__main__":
    dataset_name, data_directory, config_file_name, nb_epoch, nb_epoch_batch_size  = sys.argv[1:]
    nb_epoch = int(nb_epoch)
    nb_epoch_batch_size = int(nb_epoch_batch_size)

    optimizers = ["adam", "rmsprop"]
    decays_gen = frange(0, .5, .1)
    decays = list(decays_gen)

    for optimizer in optimizers:
        for decay in decays:
            print "----------------------", optimizer, decay, "----------------------"
            trainer = TrainFull.TrainFull(config_file_name, dataset_name, data_directory, 
                            nb_epoch, nb_epoch_batch_size, optimizer, decay)
            trainer.train()

