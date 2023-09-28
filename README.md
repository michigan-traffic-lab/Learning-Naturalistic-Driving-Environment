# Learning Naturalistic Driving Environment with Statistical Realism

[Michigan Traffic Lab, University of Michigan](https://traffic.engin.umich.edu/)

<!-- ABOUT THE PROJECT -->
# Introduction of the project

## About
This project contains the source code and data for the paper "[Learning naturalistic driving environment 
with statistical realism](https://www.nature.com/articles/s41467-023-37677-5)".

## Code structure

```
Learning-Naturalistic-Driving-Environment/
|__ basemap: visualize the simulation environment
|__ behavior_net: behavior modeling network
|__ configs: configuration files
|__ data: source data for inference and training
|__ geo_engine: coordinate transformation
|__ result_analysis: analyze experiment results and plot figures
|__ road_matching: region of interest (ROI) matching for metrics analysis
|__ sim_evaluation_metric: calculate simulation metrics
|__ simulation_modeling: simulation environment
|__ trajectory_pool: manage vehicle trajectory during simulation
|__ vehicle: vehicle class
|__ requirements.txt: required packages
|__ run_inference.py: main function for running simulations
|__ run_training_behavior_net.py: main function for training behavior modeling network
```

# Citation

If you find this work useful in your research, please consider cite:

```
@article{yan2023learning,
  title={Learning naturalistic driving environment with statistical realism},
  author={Yan, Xintao and Zou, Zhengxia and Feng, Shuo and Zhu, Haojie and Sun, Haowei and Liu, Henry X},
  journal={Nature Communications},
  volume={14},
  number={1},
  pages={2037},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```

# Installation

## Installation and configuration

### Clone this repository

```bash
git clone https://github.com/michigan-traffic-lab/Learning-Naturalistic-Driving-Environment.git
```

### Create a new virtual environment

You are recommended to create a new Conda environment to install the project
```bash
conda create -n NNDE python=3.9
```

### Install all required packages

To install the Python packages required for this repository, execute the command provided below:
```bash
conda activate NNDE
pip install -r requirements.txt
```

### Add this project to PYTHONPATH
For Anaconda users (recommended), you can use [conda-develop](https://docs.conda.io/projects/conda-build/en/latest/resources/commands/conda-develop.html) 
to add the main folder path to the specific
conda virtual environment as follows:
```bash
conda activate NNDE
conda install conda-build
conda develop .
```

Alternatively, set the main folder path (your-absolute-main-folder-path) to PYTHONPATH in system variables.

# Dataset

## Download

The data for inference (e.g., model checkpoints), training (e.g., the trajectory data for training), and analysis 
(e.g., ground truth data for metrics analysis)
can be downloaded [here](https://learning-naturalistic-driving-environment.s3.us-east-2.amazonaws.com/data.zip)(852MB).
Unzip the file and put them into the `data` folder, the file structure should look like:

```
Learning-Naturalistic-Driving-Environment/data/
|__ inference
|______ AA_rdbt
|___________ behavior_model_ckpt
|___________ safety_mapper_ckpt
|___________ ROIs-map
|___________ ...
|______ rounD
|__ paper-inference-results
|______ AA_rdbt_paper_results
|______ rounD_paper_results
|__ statistical-realism-ground-truth
|______ AA_rdbt_ground_truth
|______ rounD_ground_truth
|__ training
|______ behavior_net
|___________ AA_rdbt
```

The `inference` folder contains the model checkpoints and the other files needed for inference. 
The `paper-inference-results` folder contains the inference results for the paper.
The `statistical-realism-ground-truth` folder contains the ground truth data for metrics analysis.
The `training` folder contains the trajectory data for training the behavior modeling network.

If you do not want to re-train the model and only want to do the simulation, you will only need the `inference` and 
`statistical-realism-ground-truth` data.

## Data format

In this repository, we provide the Ann Arbor dataset that we used to train NeuralNDE in the `training` folder. 
The dataset contains the vehicle trajectory data perceived by the roadside perception system 
deployed at the two-lane roundabout at the intersection of State St. and W. Ellsworth Rd. in Ann Arbor, Michigan.

+ The data was collected from 10 am to 5 pm on May 2nd, 2022.
+ The data sample rate is 2.5Hz. 

The dataset contains multiple pickle files.

```
  |-2021-05-02 10-03-48-671927.pickle
  |-2021-05-02 10-03-49-076785.pickle
  |-2021-05-02 10-03-49-475787.pickle
  ...
```

Each pickle file contains the vehicle trajectory data of one frame. Here is an example of the information
in a pickle file

```
[
    {
        "id": "1",
        "location":
            "x": 86.76476951608201,
            "y": -25.098855800926685,
            "z": None
        "speed_heading": 163.5021444664423
        ...
    },
    {
        "id": "3",
        "location":
            "x": 69.60102021666243,
            "y": -24.766377145424485,
            "z": None
        "speed_heading": 198.9397031364029
        ...
    },
   ...
]
```

where `id` denotes the unique vehicle id, `x, y` denote the vehicle position (local x, y coordinates, unit in meter),
and `speed_heading` denotes the vehicle heading (starting from east and anticlockwise, unit in degree).

The rounD dataset is available [here](https://www.round-dataset.com/) and can be processed into the same format for 
training.

# Usage

## Inference

Config the configuration file (e.g., simulation wall time, number of simulation episodes, etc.)
in the `configs` folder
and run the following to initialize the simulator:

No visualization:
```bash
python run_inference.py --experiment-name your-experiment-name --folder-idx 1 --config ./configs/rounD_inference.yml
```
With visualization:
```bash
python run_inference.py --experiment-name your-experiment-name --folder-idx 1 --config ./configs/rounD_inference.yml --viz-flag
```

Note that in the default configuration file, we set the `use_gpu` flag to `False` to use CPU for inference. Change it
into `True` if you want to use GPU for inference.

We simulate approximately 15,000 hours of simulation to record data and generate simulated statistics to validate the 
statistical realism of the NeuralNDE, where all data are used for calculating
crash-related metrics (e.g., crash rate, type, severity) and 100 hours of data are used for other metrics.

The default time resolution is 0.4s, set `interpolate_flag` to `True` in the configuration file to interpolate the 
trajectory to a finer resolution.

## Training

### Behavior modeling network

Config the configuration file and run the following commands to train behavior modeling network.
The AA roundabout training configuration file is used as an example. The result will be saved 
in `./results/training/behavior_net` folder if you are using the default `--save-result-path`.

```bash
python run_training_behavior_net.py --config ./configs/AA_rdbt_behavior_net_training.yml --experiment-name AA_rdbt_behavior_net_training
```
Change the model path `behavior_model_ckpt_dir` in the configuration file if you want to use
newly trained behavior modeling network for inference.

> Important note: Note that training can continue from previous checkpoints. 
    When the --experiment-name and --save-result-path are the same, the code 
    will check whether this experiment has
    been conducted and continue from the last_ckpt.pt under this experiment.

Tensorboard can be used to visualize the training process (loss, etc).
Run the following code in a terminal if you are 
using the default `--save-result-path` during training, otherwise, change
the `--logdir` path to your defined save-result-path. Then, open a browser and go to http://localhost:6006. 

```bash
tensorboard --logdir=./results/training/behavior_net
```

## Data analysis

Config the settings (e.g., simulation results path) in `result_analysis/plot_all_results.py` 
and run the following to analyze simulation results and plot figures.

```bash
cd result_analysis 
python plot_all_results.py
```

Note that you need config accordingly when running the inferences to save data for the analysis.
For example, in order to generate instantaneous distributions, the 
`gen_realistic_metric_flag` and `instant_speed` flags need be set as True in your inference yml file. 
In order to generate crash type and severity results, you need to save crash data 
(i.e., set `save_collision_data_flag` as True).

# Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

# License

This project is licensed under the [PolyForm Noncommercial License 1.0.0]. Please refer to [LICENSE](https://github.com/michigan-traffic-lab/Learning-Naturalistic-Driving-Environment/blob/master/LICENSE) for more details.

# Acknowledgment

This work is supported by the U.S. Department of Transportation
Region 5 University Transportation Center: Center for Connected and Automated Transportation ([CCAT](https://ccat.umtri.umich.edu/)) 
of the University of Michigan, and [National Science Foundation](https://www.nsf.gov/).

# Developer

Xintao Yan (xintaoy@umich.edu)

Zhengxia Zou (zzhengxi@umich.edu)

# Contact

Henry Liu (henryliu@umich.edu)
