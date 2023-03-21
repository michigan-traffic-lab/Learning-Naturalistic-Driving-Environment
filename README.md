# Learning Naturalistic Driving Environment with Statistical Realism

[Michigan Traffic Lab, University of Michigan](https://traffic.engin.umich.edu/)

## Introduction

In this repository, we provide the Ann Arbor dataset that we used to train NeuralNDE.
The dataset contains the vehicle trajectory data perceived by the roadside perception system 
deployed at the two-lane roundabout at the intersection of State St. and W. Ellsworth Rd. in Ann Arbor, Michigan.

+ The data was collected from 10 am to 5 pm on May 2nd, 2022.
+ The data sample rate is 2.5Hz. 

## Data format

The dataset contains multiple json files.

```
trajectory_data
  |-2021-05-02 10-03-48-671927.json
  |-2021-05-02 10-03-49-076785.json
  |-2021-05-02 10-03-49-475787.json
  ...
```

Each json file contains the vehicle trajectory data of one frame. Here is an example of the data in a json file

```
[
    {
        "vid": "1",
        "x": 86.76476951608201,
        "y": -25.098855800926685,
        "heading": 163.5021444664423
    },
    {
        "vid": "3",
        "x": 69.60102021666243,
        "y": -24.766377145424485,
        "heading": 198.9397031364029
    },
   ...
]
```

where `vid` denotes the unique vehicle id, `x, y` denote the vehicle position (local x, y coordinates, unit in meter),
and `heading` denotes the vehicle heading (starting from east and anticlockwise, unit in degree). An illustration figure
of the roundabout is included in the zip file.

## Download

Please download the data [here](https://aa-trajectory-data.s3.us-east-2.amazonaws.com/AA-trajectory-data.zip).

## Terms of use

### License

This project is licensed under the [PolyForm Noncommercial License 1.0.0]. Please refer to [LICENSE](https://github.com/michigan-traffic-lab/Learning-Naturalistic-Driving-Environment/blob/master/LICENSE) for more details.

## Acknowledgment

This work is supported by the U.S. Department of Transportation
Region 5 University Transportation Center: Center for Connected and Automated Transportation ([CCAT](https://ccat.umtri.umich.edu/)) 
of the University of Michigan, and [National Science Foundation](https://www.nsf.gov/).

## Developers

Zhengxia Zou (zhengxiazou@gmail.com)

Xintao Yan (xintaoy@umich.edu)

## Contact

Henry Liu (henryliu@umich.edu)