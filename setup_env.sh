#!/bin/bash

# Ensure working dir is at the location of this script
cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null

# Ensure submodules are pulled
git submodule update --init

# This install was tested with cuda 12.1
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# To reset the environment in case this fails:
# conda deactivate; conda remove -n gtavatar --all -y

ENV_NAME=gtavatar
conda create -n $ENV_NAME python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME
if echo $CONDA_PREFIX | grep $ENV_NAME; then
    echo "Conda environment successfully activated"
else
    echo "Conda environment not activated. Creation most likely failed."
    exit
fi

pip install torch==2.2 torchvision==0.17 --index-url https://download.pytorch.org/whl/cu121
pip install fvcore iopath
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt221/download.html
pip install "numpy<2" # downgrade numpy

# Check:
python -c "import torch; free, total = torch.cuda.mem_get_info(); print(f'PyTorch v{torch.__version__}\nCuda? {torch.cuda.is_available()}\nGPU: {torch.cuda.get_device_name(torch.cuda.current_device())} (memory usage: {1-free/total:.2%})')"
python -c "from pytorch3d.io import load_obj; print('p3d ok')"

pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast/
pip install --no-build-isolation git+https://github.com/camenduru/simple-knn.git
pip install --no-build-isolation git+https://github.com/KelianB/diff-surfel-rasterization-uv-tex.git

pip install mediapipe==0.10.33
pip install configargparse ninja lpips roma natsort gdown scikit-image==0.25.2 imageio==2.37.3 opencv-python-headless==4.13.0.92 timm~=0.9.16
pip install --no-build-isolation git+https://github.com/mattloper/chumpy # see https://github.com/mattloper/chumpy/issues/49

# For interact.py only
pip install pyqt5==5.15.11

pip install "numpy<2" # downgrade numpy

# Note that pip warnings about some dependencies requiring numpy>=2 are expected, but should not cause issues. 
