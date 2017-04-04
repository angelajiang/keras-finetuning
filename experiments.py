import FineTunerFast as ft
import sys

def accuracy_per_layer(config_file, dataset_dir, output_file, model_prefix, max_layers, layers_stride):
    ft_obj = ft.FineTunerFast(config_file, dataset_dir, model_prefix)
    f = open(output_file, 'w', 0)
    for num_training_layers in range(layers_stride, max_layers + layers_stride, layers_stride):
        print "[experiments] ================= Finetunning", num_training_layers, "layers ================= "
        acc = ft_obj.finetune(num_training_layers)
        acc = str.format("{0:.4f}", acc)
        line = str(num_training_layers) + "," + str(acc) + "\n"
        print "[RESULT] Num fine-tuned layers: ", num_training_layers, " accuracy: ", acc
        f.write(line)
        f.flush()
    ft_obj.print_config()


if __name__ == "__main__":
    config_file, dataset_dir, output_file, model_prefix, max_layers, layers_stride = sys.argv[1:]
    max_layers = int(max_layers)
    layers_stride = int(layers_stride)
    accuracy_per_layer(config_file, dataset_dir, output_file, model_prefix, max_layers, layers_stride)

