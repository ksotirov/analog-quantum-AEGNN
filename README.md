# Analog Quantum Asynchronous Event-Based Graph Neural Network
This repository contains code for "Analog Quantum Asynchronous Event-Based Graph Neural Network" - Kristian Sotirov, Shaheen Acheche, Antonio A. Gentile, and Osvaldo Simeone

## Dependencies

The program is written with Python 3.12.3. The code is executed on an Ubuntu 24.04.4 LTS operating system. 

To recreate the environment, a virtual environment (venv) needs to be created. To create the environment, execute the following command in the terminal:
```
python -m venv venv
source venv/bin/activate
```

The second command assures we are running in the newly created environment. Once inside the environment, install the dependencies by executing in the terminal:

```
pip install -r requirements.txt
```

## Basic Usage

 - Main file is `neuromorphic_qek_experiment.py`. The file runs a series of experiments using the QA-AEGNN algorithm on the synthetic dataset task. The script uses the arguments provided in the command line to specify the hyperparameters of the system. If nothing is specified, the scripts use default values. For more information regarding the command line arguments:

```
neuromorphic_qek_experiment.py --help
```

The script outputs `results/neuromorphic_qaaegnn_<RUN_NUMBER>'.csv`, where `RUN_NUMBER` is specified by `--run_number` argument when executing the script. Default `RUN_NUMBER` is 0.  


## Specify noise model

To specify noise model, use `--allow_noise` argument needs to be specified. The script expects a JSON file, to describe a noise model. The JSON string must contain `noise_types` parameter, describing the types of simulated quantum noise (i.e., SPAM, dephasing, relaxation) in an array, followed by names and values of the specified parameters.

