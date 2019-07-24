import os
from models.setup import Setup
from core.config_loader import SysConfig


def main():

    setup = Setup.load()

    if setup.dataset_id is None:

        for tf in [True, False]:
            setup.is_minute_feature = tf

            setup.dataset_id = 'sm_cum_odds'
            setup.features = 'cum_scouts'
            create_file(setup)

            setup.dataset_id = 'sm_cum_odds'
            setup.features = 'cum_goals'
            create_file(setup)

            setup.dataset_id = 'sm_comp_odds'
            setup.features = 'comp_scouts'
            create_file(setup)

            setup.dataset_id = 'sm_comp_odds'
            setup.features = 'comp_goals'
            create_file(setup)

    else:
        create_file(setup)


def create_file(setup):

    commands = list()

    if setup.is_minute_feature:

        job_filename = '_'.join([setup.classifier, setup.virtualenv, setup.dataset_id, setup.features, 'ALL.sh'])
        label = '_'.join([setup.classifier, '0', setup.dataset_id, setup.features])
        command = get_prefix_command(label, setup)
        command = add_default_parameters(setup, command)
        command += ' -tf '
        commands.append(command)

    else:

        job_filename = '_'.join([setup.classifier, setup.virtualenv, setup.dataset_id, setup.features, 'MIN.sh'])
        for minute in range(setup.minute):
            label = '_'.join([setup.classifier, str(minute), setup.dataset_id, setup.features])

            command = get_prefix_command(label, setup)
            command = add_default_parameters(setup, command)

            command += ' -t ' + str(minute)
            commands.append(command)

    jobs_dir = SysConfig.path('jobs')
    job_file = os.path.join(jobs_dir, job_filename)
    with open(job_file, 'w') as outfile:
        outfile.write('#!/usr/bin/env bash')
        outfile.write('\n')
        for c in commands:
            outfile.write(c)
            outfile.write('\n')


def get_prefix_command(label, setup):

    command = 'sbatch --job-name ' + label
    command += ' --output logs/%j-' + label + '.out'
    command += ' --error logs/%j-' + label + '.err'
    command += ' ./standard/' + setup.virtualenv + '.sh '
    return command


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