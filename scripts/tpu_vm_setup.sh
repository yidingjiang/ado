#!/bin/sh

sudo apt update
sudo apt install -y python3-venv

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install jax for TPU
pip install jax[tpu]==0.4.28 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

cd ado
pip install -r requirements.txt

# Clone and install lm-evaluation-harness into home directory
cd $HOME
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
cd ..

# Deactivate virtual environment
deactivate

echo "Setup complete. To activate the virtual environment, run: source venv/bin/activate"