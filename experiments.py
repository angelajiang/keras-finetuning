import FineTuner as ft
import sys

def accuracy_per_layer(config_file, dataset_dir, model_prefix, max_layers, layers_stride):
    ft_obj = ft.FineTuner(config_file, dataset_dir, model_prefix)
    for num_training_layers in range(0, max_layers, layers_stride):
        acc = ft_obj.finetune(num_training_layers)
        print num_training_layers, acc
    ft_obj.print_config()

if __name__ == "__main__":
    config_file, dataset_dir, model_prefix, max_layers, layers_stride = sys.argv[1:]
    max_layers = int(max_layers)
    layers_stride = int(layers_stride)
    accuracy_per_layer(config_file, dataset_dir, model_prefix, max_layers, layers_stride)

