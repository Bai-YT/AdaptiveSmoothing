import os
import click

import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

from adaptive_smoothing.trainer import CompModelTrainer
from adaptive_smoothing.utils import seed_all, read_yaml, eval_configs


def process_config(config, training):
    # Process config file (batch size)
    config['training_settings']['batch_size_test'] = (
        config['training_settings']['batch_size_test_per_gpu'] * torch.cuda.device_count())
    config['training_settings']['batch_size_train'] = (
        config['training_settings']['batch_size_train_per_gpu'] * torch.cuda.device_count())
    config['pgd_settings']['aa_batch_size'] = (
        config['training_settings']['batch_size_test'] * 5 // config['dataset_settings']['blocks'])

    # Process config file (attacks to run)
    if config['pgd_settings']["attacks_to_run"] == 'default':
        config['pgd_settings']["attacks_to_run"] = (
            ['apgd-ce', 'apgd-t'] if training else ['apgd-ce', 'apgd-t', 'fab-t', 'square'])

    # Process config file (dataset path and attack radius)
    root_path = os.path.dirname(os.getcwd())  # The path where the "Dataset" directory resides
    config['dataset_settings']['path'] = os.path.join(root_path, config['dataset_settings']['path'])
    if config['pgd_settings']['eps'] == 'default':
        config['pgd_settings']['eps'] = 8. / 255. if config['pgd_settings']['type'] == 'l_inf' else .5


@click.command(context_settings={'show_default': True})
@click.option('--training/--eval', required=True, show_default=True)
@click.option('--config', required=True, show_default=True)
@click.option('--debug', is_flag=True, default=False, show_default=True)
def run(training, config, debug):
    # Load config file
    config = read_yaml(config)
    eval_configs(config)

    # Load paths
    log_path = os.path.join(config['log_dir'], config['log_path'])
    load_path = config['model_load_path']['comp']
    load_path = os.path.join(config['log_dir'], load_path) if load_path is not None else load_path
    
    # Preprocess config file
    process_config(config, training)

    # Reproducibility
    seed_all(config['seed'])

    # Experiment driver code
    comp_trainer = CompModelTrainer(
        std_load_path=os.path.join(config['log_dir'], config['model_load_path']['std']), 
        rob_load_path=os.path.join(config['log_dir'], config['model_load_path']['rob']),
        forward_settings=config['forward_settings'], 
        dataset_settings=config['dataset_settings'], 
        training_settings=config['training_settings'], 
        pgd_settings=config['pgd_settings'], 
        randomized_attack_settings=config['randomized_attack_settings'])

    comp_trainer._comp_model.enable_autocast = training
    if training:  # Training
        comp_trainer.train(
            save_path=log_path, load_path=load_path,
            eval_freq=config['training_settings']['eval_freq'], debug=debug
        )
    else:  # Evaluating
        comp_trainer.evaluate(
            load_path=load_path, save_path=log_path, 
            auto_attack=config['pgd_settings']['use_aa'], full=False
        )
        # comp_trainer.evaluate_adap(clean_only=True, full=True)
        # comp_trainer.evaluate_data()


if __name__ == "__main__":
    run()
