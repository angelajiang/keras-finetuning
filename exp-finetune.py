import ConfigParser
import FineTunerFast as ft
import sys

def accuracy_per_layer(config_file, dataset_name, dataset_dir, model_prefix, max_layers, layers_stride):
    ft_obj = ft.FineTunerFast(config_file, dataset_dir, model_prefix)
    config_parserr = ConfigParser.RawConfigParser()   
    config_parserr.read(config_file)
    nb_epoch = str(config_parserr.get('finetune-config', 'nb_epoch'))
    optimizer_name = str(config_parserr.get('finetune-config', 'optimizer'))
    decay = str(config_parserr.get('finetune-config', 'decay'))
    lr = str(config_parserr.get('finetune-config', 'learning-rate'))
    output_dir = str(config_parserr.get('finetune-config', 'output_dir'))
    data_augmentation = bool(int(config_parserr.get('finetune-config', 'data_augmentation')))
    if (data_augmentation):
        output_file = output_dir + dataset_name + "-epochs" + nb_epoch + "-" + optimizer_name + "-decay" + decay + "-lr" + lr + "-data_aug-" + str(max_layers) + ":" + str(layers_stride)
    else:
        output_file = output_dir + dataset_name + "-epochs" + nb_epoch + "-" + optimizer_name + "-decay" + decay + "-lr" + lr + "-" + str(max_layers) + ":" + str(layers_stride)
    print output_file
    f = open(output_file, 'w', 0)
    for num_training_layers in range(layers_stride, max_layers + layers_stride, layers_stride):
        print "[experiments] ================= Finetunning", num_training_layers, "layers ================= "
        acc = ft_obj.finetune(num_training_layers)
        acc = str.format("{0:.4f}", acc)
        line = str(num_training_layers) + "," + str(acc) + "\n"
        print "[RESULT] Num fine-tuned layers: ", num_training_layers, " accuracy: ", acc
        f.write(line)
        f.flush()
    f.close()
    ft_obj.print_config()

if __name__ == "__main__":
    config_file, dataset_name, dataset_dir, model_prefix, max_layers, layers_stride = sys.argv[1:]
    max_layers = int(max_layers)
    layers_stride = int(layers_stride)
    accuracy_per_layer(config_file, dataset_name, dataset_dir, model_prefix, max_layers, layers_stride)

