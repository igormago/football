import os
from models import utils
from core.config_loader import SysConfig


def main():

    setup = utils.get_setup()
    commands = list()

    for minute in range(setup.minute):

        label = '_'.join([setup.classifier, str(minute), setup.dataset_id, setup.features])

        command = 'sbatch --job-name ' + label
        command += ' --output logs/%j-' + label + '.out'
        command += ' --error logs/%j-' + label + '.err'
        command += ' ./standard/' + setup.virtualenv + '.sh'

        command = add_default_parameters(setup, command)

        command += ' -t ' + str(minute)
        commands.append(command)

    if setup.is_minute_feature:

        command = ''
        command = add_default_parameters(setup, command)
        command += ' -tf '
        commands.append(command)

    jobs_dir = SysConfig.path('jobs')
    job_filename = '_'.join([setup.classifier, setup.dataset_id, setup.features, '.sh'])
    job_file = os.path.join(jobs_dir, job_filename)

    with open(job_file, 'w') as outfile:
        for c in commands:
            outfile.write(c)
            outfile.write('\n')

    print(commands)


def add_default_parameters(config, command):

    command += ' -c ' + config.classifier
    command += ' -d ' + config.dataset_id
    command += ' -f ' + config.features

    if config.is_to_tune:
        command += ' -au'
    if config.is_to_train:
        command += ' -ar'
    if config.grid_type:
        command += ' -gt ' + str(config.grid_type)
    if config.grid_params:
        command += ' -gp ' + str(config.grid_params)
    if config.train_size:
        command += ' -ts ' + str(config.train_size)

    return command


if __name__ == "__main__":
    main()