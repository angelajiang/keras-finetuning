import sys
sys.path.append('../util')
import DataSet
import inference
import pprint as pp

def run(dataset_dir, batch_size, num_apps_range, num_apps_stride, output_dir):
    xs = [x for x in range(1, num_apps_range + 1, num_apps_stride)]
    data = DataSet.DataSet(dataset_dir, 224)
    shared = []
    not_shared = []
    for num_apps in xs:
        fps = inference.benchmark(data, num_apps, batch_size, False)
        not_shared.append(fps)
        fps = inference.benchmark(data, num_apps, batch_size, True)
        shared.append(fps)
    print "[OUTPUT] Shared:", shared
    print "[OUTPUT] Not shared:", not_shared
    output_file = output_dir + "/" + "toplayer-bs" + str(batch_size) + "-" + str(num_apps_range) + "-" + str(num_apps_stride)
    with open(output_file, "w") as f:
        for x, ns, s in zip(xs, not_shared, shared):
            ns = str.format("{0:.4f}", ns)
            s = str.format("{0:.4f}", s)
            line = str(x) + "," + str(ns) + "," + str(s) + "\n"
            f.write(line)

def get_layer_runtime(dataset_dir, batch_size, output_dir):
    data = DataSet.DataSet(dataset_dir, 224)
    trials = 100
    runtime_by_num_layers = inference.benchmark_runtime(data, batch_size, trials)
    output_file = output_dir + "/" + "runtimes-bs" + str(batch_size) + "-t" + str(trials)
    with open(output_file, "w") as f:
        for k, v in runtime_by_num_layers.iteritems():
            v = str.format("{0:.4f}", v)
            line = str(k) + "," + str(v) + "\n"
            f.write(line)
    print output_file

if __name__ == "__main__":
    dataset_dir, batch_size, num_apps_range, num_apps_stride, output_dir = sys.argv[1:]
    #run(dataset_dir, int(batch_size), int(num_apps_range), int(num_apps_stride), output_dir)
    get_layer_runtime(dataset_dir, int(batch_size), output_dir)
