#!/bin/bash

git clone https://github.com/li-group/SafeOR-Gym.git
cd SafeOR-Gym

# Create and activate conda environment from environment file
conda env create --name safe_rl_env --file environment.yml
conda activate safe_rl_env

pip install -r requirements.txt

cd omnisafe
pip install -e .
cd ..