import FineTuner as ft
import sys

def accuracy_per_layer(config_file, dataset_dir, output_file, model_prefix, max_layers, layers_stride):
    ft_obj = ft.FineTuner(config_file, dataset_dir, model_prefix)
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

def accuracy_per_model(config_file, dataset_dir, model_prefix):
    ft_obj = ft.FineTuner(config_file, dataset_dir, model_prefix)
    acc = ft_obj.evaluate(ft_obj.model)
    print acc

def test_determinism(config_file, dataset_dir, model_prefix, num_trials, num_training_layers):
    ft_obj = ft.FineTuner(config_file, dataset_dir, model_prefix)
    for i in range(num_trials):
        acc = ft_obj.finetune(num_training_layers)
        print "===========", i, ",", acc, "==========="

if __name__ == "__main__":
    config_file, dataset_dir, output_file, model_prefix, max_layers, layers_stride = sys.argv[1:]
    max_layers = int(max_layers)
    layers_stride = int(layers_stride)
    accuracy_per_layer(config_file, dataset_dir, output_file, model_prefix, max_layers, layers_stride)
    #accuracy_per_model(config_file, dataset_dir, model_prefix)
    #test_determinism(config_file, dataset_dir, model_prefix, 3, 100)

