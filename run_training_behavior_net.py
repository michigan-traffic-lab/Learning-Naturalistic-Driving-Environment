import argparse
import os
import yaml
import shutil
import matplotlib.pyplot as plt

from behavior_net import datasets
from behavior_net import Trainer

# settings
parser = argparse.ArgumentParser()
parser.add_argument('--experiment-name', type=str, required=True,
                    help='The name of the experiment folder where the data will be stored')
parser.add_argument('--save-result-path', type=str, default=r'./results/training/behavior_net',
                    help='The path to save the training results, a folder with experiment_name will be created in the path')
parser.add_argument('--config', type=str, required=True,
                    help='The path to the training config file. E.g., ./configs/AA_rdbt_behavior_net_training.yml')
args = parser.parse_args()


def check_data_loading():
    # to check if the data is loading correctly?
    for i in range(100):
        data = next(iter(dataloaders['train']))
        x_in = data['input'][0]
        x_out = data['gt'][0]
        plt.subplot(1, 2, 1), plt.imshow(x_in), plt.title('input')
        plt.subplot(1, 2, 2), plt.imshow(x_out), plt.title('gt')
        plt.show()
        print(data['input'].shape)
        print(data['gt'].shape)


if __name__ == '__main__':
    # Load config file
    with open(args.config) as file:
        try:
            configs = yaml.safe_load(file)
            print("Loading config file: {0}".format(args.config))
        except yaml.YAMLError as exception:
            print(exception)

    # Checkpoints and training process visualizations save paths
    experiment_name = args.experiment_name
    save_result_path = args.save_result_path
    configs["checkpoint_dir"] = os.path.join(save_result_path, experiment_name, "checkpoints")  # The path to save trained checkpoints
    configs["vis_dir"] = os.path.join(save_result_path, experiment_name, "vis_training")  # The path to save training visualizations

    # Save the config file of this experiment
    os.makedirs(os.path.join(save_result_path, experiment_name), exist_ok=True)
    save_path = os.path.join(save_result_path, experiment_name, "config.yml")
    shutil.copyfile(args.config, save_path)

    # Initialize the DataLoader
    dataloaders = datasets.get_loaders(configs)

    # Check data loading
    # check_data_loading()

    # Initialize the training process
    m = Trainer(configs=configs, dataloaders=dataloaders)
    m.train_models()
