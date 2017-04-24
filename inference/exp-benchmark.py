import inference
import sys

def run(dataset_dir, batch_size, num_apps_range, num_apps_stride, output_dir):
    xs = [x for x in range(1, num_apps_range + 1, num_apps_stride)]
    shared = []
    not_shared = []
    try:
        for num_apps in xs:
            fps = inference.benchmark(dataset_dir, num_apps, batch_size, True)
            shared.append(fps)
        for num_apps in xs:
            fps = inference.benchmark(dataset_dir, num_apps, batch_size, False)
            not_shared.append(fps)
    except:
        total_completed = max(len(shared), len(not_shared))
        xs = xs[:total_completed]
        shared = shared[:total_completed]
        not_shared = not_shared[:total_completed]
    print "[OUTPUT] Shared:", shared
    print "[OUTPUT] Not shared:", not_shared
    output_file = output_dir + "/" + "toplayer-bs" + str(batch_size) + "-" + str(num_apps_range) + "-" + str(num_apps_stride)
    with open(output_file, "w") as f:
        for x, ns, s in zip(xs, not_shared, shared):
            ns = str.format("{0:.4f}", ns)
            s = str.format("{0:.4f}", s)
            line = str(x) + "," + str(ns) + "," + str(s) + "\n"
            f.write(line)



if __name__ == "__main__":
    dataset_dir, batch_size, num_apps_range, num_apps_stride, output_dir = sys.argv[1:]
    run(dataset_dir, int(batch_size), int(num_apps_range), int(num_apps_stride), output_dir)
