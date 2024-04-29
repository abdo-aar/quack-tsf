# QuaCK-TSF

This repository contains the official implementation for the paper _**QuaCK-TSF: Quantum-Classical Kernelized Time
Series Forecasting**_.

## Project structure


```
quack-tsf/
│
├── data/                       # Data directory
│   ├── synthetic/              # Synthetic, generated data files
│   └── real/                   # Real time sereis data files
│
├── notebooks/                  # Jupyter notebooks for exploration and presentation
│
├── src/                        # Source code for the project
│   ├── __init__.py             # Makes src a Python module
│   ├── data/                   # Scripts to download or generate data
│   │   ├── synthetic/          # Scripts to generate/ handle synthetic ts data
│   │   └── real/               # Scripts to download/ handle real ts data
│   ├── models/                 # Scripts that defines gp models
│   │   ├── classical_kernels/  # Scripts to implement classical kernel models
│   │   ├── gp_models/          # Scripts to handle gp kernel models
│   │   └── quantum_kernels/    # Scripts to implement quantum kernel models
│   ├── optimization/           # Scripts used to train gp models using a Bayesian Optimization process
│   │   ├── bo_optim_utils/     # Scripts that hold utility functions used during the BO process
│   │   ├── configs/            # Config files that define the parametrization of the GP model and its fitting procedure
│   │   │   ├── bo_optim/       # Configs holding the BO process configuration to use
│   │   │   ├── data/           # Configs holding the data to use configuration
│   │   │   ├── model/          # Configs holding the gp model to use configuration
│   │   │   └── config.yaml     # Main config file
│   │   └── bo_process.py       # Script that trains a gp model using a BO process  
│   └── visualization/          # Scripts to create exploratory and results oriented visualizations
│
├── storage/                    # Experiments' results. Directory generated automatically, and not handled by Git 
│
├── tests/                      # Test cases to ensure your code behaves as expected
│
├── requirements.txt            # The dependencies file for reproducing the analysis environment
│
├── setup.py                    # Makes project pip installable (pip install -e .) so src can be imported
├── Dockerfile                  # Sets up the docker image
├── environment.yml             # The environment libraries used in the project
└── README.md                   # The top-level README for developers using this project
```

<!-- Utility commands -->
<!-- Export python path: ``export PYTHONPATH=${PYTHONPATH}:${pwd} ``-->
<!-- Run jupyter-lab server ``jupyter lab --ip 10.44.83.233 --port 8899 --no-browser`` -->

## Downloading existing experiments 
To download experiments already conducted, and that are included in the paper results, use the following:
1. 

## Training GP models
### To train a single GP model use the command:
``python src/optimization/main.py   bo_optim=optim data=synthetic model=iqp_kernel # Provided that the config file is well set``

### To continue an interrupted GP model training use the command:
``python src/optimization/main.py  --experimental-rerun path_to_exp_dir/.hydra/config.pickle``
<!-- python src/optimization/main.py  --experimental-rerun storage/experiments/iqp_kernel/2024-04-02/15-13-26/.hydra/config.pickle -->

### To train multiple GP models using different window_length hparam use the command:
``python src/optimization/main.py -m bo_optim=optim data=synthetic model=iqp_kernel bo_optim.train_hparams.window_length=4,5,6,7,8,9,10 hydra/launcher=joblib``

### Run on IBMQ machines
 ``python src/optimization/main_qHardware.py   bo_optim=optim data=synthetic model=iqp_kernel_qq``


## To use Docker
### Build the Docker image 
``docker build --network=host -t quantique/quack-tsf .``

### Make sure you're host machine is configured to allow docker use GPU acceleration power
If not, configure it on the host machine. For a Ubuntu distributed host machine example here are the steps 
for such configuration: 
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Run a container interactively
``docker run -it --gpus all --network=host --name quack-tsf-container quantique/quack-tsf``

### Run the scripts in the previous section (Training GP models) inside the docker container

scp aara2601@dinf-attagis:/home/local/USHERBROOKE/aara2601/projects/quack-tsf/experiments.zip C:\Users\aara2601\Downloads