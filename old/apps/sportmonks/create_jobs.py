from core.config import PATH_JOBS

def cancel_jobs():
    a = []
    for i in range(193612, 193612+41):
        print("scancel " + str(i))


def trainning_cpu(classifier):

    for i in range(0, 96):
        for j in ['S', 'G']:
            print("sbatch --job-name CIKM-" + str(i) + "-" + str(j) + " trainning_cpu.sh "
                  + classifier + " " + str(i) + " " + j)


def testing_cpu(file_name, classifier):

    file_jobs = PATH_JOBS + classifier + "_" + file_name
    f = open(file_jobs, "w")
    f.write("#!/usr/bin/env bash \n")
    for minute in range(0, 96):
        for group in ['S', 'G']:
            label = classifier + "-" + str(minute) + "-" + str(group)
            f.write("sbatch --job-name " + label + " --output logs/%j-" + label + ".out --error logs/%j-"
                  + label + ".err " + file_name + " " + classifier + " " + str(minute) + " " + group + "\n")

    print(f)
    f.close()

def tunning_cpu(file_name, classifier, step):

    file_jobs = PATH_JOBS + classifier + "_" + file_name
    f = open(file_jobs, "w")
    f.write("#!/usr/bin/env bash \n")
    for minute in range(0, 96):
        for group in ['S', 'G']:
            label = classifier + "-" + str(minute) + "-" + str(group) + "-" + str(step)
            f.write("sbatch --job-name " + label + " --output logs/%j-" + label + ".out --error logs/%j-"
                    + label + ".err " + file_name + " " + classifier + " " + str(minute) + " " + group + " " +
                    str(step) + "\n")

    print(f)
    f.close()

def tunning_mlp(file_name, classifier, method):

    config = dict()
    config['classifier']= 'mlp'
    config['minute'] = 45
    config['group'] = 'S'
    config['cnn_encoder_layer_sizes'] = 4
    config['optim_learning_rate'] = 0.00001
    config['optim_num_epochs'] = 100
    config['optim_batch_size'] = 32
    config['method'] = 'dense_1_layer_d'
    config['optim:dropout_rate'] = 0
    config['step']=1

    label = str()
    for c in config.keys():
        label = label + str(config[c]) + "_"

    file_jobs = PATH_JOBS + label + ".sh"

    f = open(file_jobs, "w")
    f.write("#!/usr/bin/env bash \n")

    command = str()
    for c in config.keys():
        command = command + c + "=" + str(config[c]) + " "

    line = "sbatch --job-name " + label + \
           " --output logs/%j-" + label + \
           ".out --error logs/%j-" + label + ".err " + file_name + " " \
           + command + "\n"

    print(line)
    f.write(line)

    f.close()


def test():
    for i in range(0,96):
        line = 'sbatch --job-name mlp_45_S_8_1e-05_100_32_dense_1_layer_1_ --output logs/%j-mlp_45_S_8_1e-05_100_32_dense_1_layer_1_.out --error logs/%j-mlp_45_S_8_1e-05_100_32_dense_1_layer_1_.err tunning_gpu.sh classifier=dnn minute=' + str(i) + ' group=S cnn_encoder_layer_sizes=5,4 optim_learning_rate=1e-04 optim_num_epochs=500 optim_batch_size=256 method=mlst_fcn step=1 optim:dropout_rate=0'
        print(line)

def main():
    tunning_mlp('tunning_gpu.sh', 'mlp', method='dense_1_layer')

if __name__ == "__main__":
    test()