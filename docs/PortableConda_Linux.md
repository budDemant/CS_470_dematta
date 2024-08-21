# Setup for Portable Conda Development Environment - Linux

## Initial Miniconda Setup

Miniconda is a smaller version of Anaconda, which is a distribution of Python.

First, download the latest version:
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```

Next, install it into your home directory (default options are fine):
```
~/miniconda3/bin/conda init bash
```

Close and reopen the terminal; you should see ```(base)``` to the left of your terminal prompt.

Create your CV environment with Python 3.10:
```
conda create -n CV python=3.10
```

Before installing any packages to the new environment, activate it:
```
conda activate CV
```

```(CV)``` should now be to the left of your terminal prompt.


## Installing CUDA
Install the necessary CUDA toolkit and CUDNN libraries:
```
conda install -y -c conda-forge cudatoolkit=11.8.0
pip install nvidia-cudnn-cu11==8.6.0.163
conda install -y -c nvidia cuda-nvcc=11.3.58
```

## Installing Pytorch
To install Pytorch that works with CUDA 11.8:
```
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
To verify Pytorch works correctly:
```
python3 -c "import torch;x=torch.rand(5, 3);print(x);print(torch.cuda.is_available())"
```
You should see an array printed and the word ```True```.

## Installing TensorFlow
Install CMake and Lit using conda:
```
conda install -y cmake lit
```
Then, install TensorFlow 2.13:
```
pip install tensorflow==2.13.*
```
Set up necessary paths so CUDA and CUDNN libraries can be found:
<sub><sup>
```
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' \
>> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
printf 'export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib/\n' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```
</sup></sub>

The ```libdevice``` file should also be copied (particularly relevant for Ubuntu 22.04):

```
mkdir -p $CONDA_PREFIX/lib/nvvm/libdevice
cp $CONDA_PREFIX/lib/libdevice.10.bc $CONDA_PREFIX/lib/nvvm/libdevice/
```

To verify that TensorFlow is working properly:
```
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
This should list GPU device(s) (similar to the following):
```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```



## Installing Other Python Packages
The other Python packages you should install via conda:
```
conda install -y -c pytorch -c nvidia -c conda-forge -c defaults pandas scikit-learn scikit-image matplotlib pylint gradio
```
...and the ones to install via pip:
```
pip install opencv-python
```


## Making a Portable Environment

While your CV environment is active, install the ```conda-pack``` tool:
```
conda install -c conda-forge conda-pack
```
Then, create your portable environment using conda-pack:
```
conda-pack
```
This will create ```CV.tar.gz```.

Create a folder on your USB drive and copy ```CV.tar.gz``` to this folder:
```
export MY_USB=/media/realemj/BAT_DRIVE/CV
mkdir $MY_USB
cp CV.tar.gz $MY_USB/CV.tar.gz
```
Change to this new directory and extract the files
```
cd $MY_USB
tar -xvzf CV.tar.gz
```

**You MAY also have to manually copy ```etc\conda\activate.d\env_vars.sh``` to the corresponding folder on your USB drive!**

**PLEASE NOTE**: If the machine you are running this environment on does NOT have conda installed, you will NOT be able to install additional packages via conda (since the portable environment does not include the conda tool).

To activate your new environment on another machine:
```
export MY_USB=/media/realemj/BAT_DRIVE/CV
source $MY_USB/bin/activate
```

## Portable Visual Code
Go [here](https://code.visualstudio.com/sha/download?build=stable&os=linux-x64) and download the tar.gz version of Visual Code.
Unpack it to your USB drive.  Open a terminal inside this folder (e.g., VS-Linux-x64) and create a ```data``` folder:
```
mkdir data
```
This will cause Visual Code to store extensions and settings locally.

To run Visual Code, while in the VS-Linux-x64 folder, type:
```
./bin/code
```

Install the following extensions:
- **Python Extension Pack** by Don Jayamanne
- **Git Graph** by mhutchie

A terminal can always be created/opened with ```View menu -> Terminal```.  However, if you need to restart, click the garbage can icon on the terminal window to destroy it.

Once you open a project, to make sure you are using the correct Python interpreter:
1. Close any open terminals with the garbage can icon
2. Open a .py file
3. View -> Command Palette -> "Python: Select interpreter"
4. Choose the one located at CV/bin/python in your USB drive
5. If the GPU is not being used, type the following in your terminal to manually activate your environment: ```source $MY_USB/bin/activate```, where ```$MY_USB``` is the location of your CV folder

## Troubleshooting

* If you need to remove the CV environment (the LOCAL version, not the portable one):
```conda remove --name CV --all```

* If you try to do ```conda-pack```, and it gives you errors about conflicts between conda-managed packages and pip-managed packages, you will want to uninstall the package via pip and force install via conda.  The two common problems are:
    * typing_extensions: 4.7.1
    * numpy-base: 1.25.2
```
pip uninstall typing_extensions numpy-base
conda install -f typing_extensions==4.7.1 numpy-base==1.25.2
```
* If Intellisense is highlighting OpenCV (cv2) commands red:
    1. Open ```File menu -> Preferences -> Settings```
    2. Select the ```User``` settings tab
    3. Search for ```pylint```
    4. Under ```Pylint:Args```, add an item: ```--generate-members```

* If you are unable to download datasets (specifically if you get SSL-related errors), try adding the following to the top of your code file:
```
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

* If conda is not installed on the machine you are running the environment on, Visual Code will not activate the environment correctly.  Either 1) install Miniconda3 ONLY on your machine, or 2) in a terminal in Visual Code, run the following (replacing the value of ```MY_USB``` with the correct path to your CV environment on your USB drive):
```
export MY_USB=/media/realemj/BAT_DRIVE/CV
source $MY_USB/bin/activate
```