# Setup for Portable Conda Development Environment - Windows

## Initial Miniconda Setup

Miniconda is a smaller version of Anaconda, which is a distribution of Python.

First, download the latest installer for Windows [here](https://docs.conda.io/projects/miniconda/en/latest/).

Run the installer; I would install it for All Users (especially if your username has spaces in it).  I installed it into ```C:/miniconda3```.

Open "Anaconda Prompt (miniconda3)" **as an administrator**; you should see ```(base)``` to the left of your terminal prompt.

Create your CVwin environment with Python 3.10:
```
conda create -n CVwin python=3.10
```

Before installing any packages to the new environment, activate it:
```
conda activate CVwin
```

```(CVwin)``` should now be to the left of your terminal prompt.

## Installing CUDA
Install the necessary CUDA toolkit and CUDNN libraries:
```
conda install -y -c conda-forge cudatoolkit=11.8.0
pip install nvidia-cudnn-cu11==8.9.4.25
```

## Installing Pytorch
To install Pytorch that works with CUDA 11.8:
```
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
To verify Pytorch works correctly:
```
python -c "import torch; x = torch.rand(5, 3); print(x); print(torch.cuda.is_available())"
```
You should see an array printed and the word ```True```.

## Installing TensorFlow
Install CMake and Lit using conda:
```
conda install -y cmake lit
```
Then, install TensorFlow 2.10 (OTHERWISE GPU WILL NOT WORK):
```
pip install "tensorflow<2.11"
```
Now we need to set up the necessary paths so CUDA and CUDNN libraries can be found.  Your environment should be located at ```C:\miniconda3\envs\CVwin```.  Inside this folder, under ```etc\conda\activate.d```, create a file ```env_vars.bat```.  If the folder path does not exist under ```CVwin```, create it as well.  If you do not have the permissions, create the file elsewhere and copy it in.  Inside this file:
```
@echo off
for /f "delims=" %%a in ('python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"') do @set CUDNN_FILE=%%a
for %%F in ("%CUDNN_FILE%") do set CUDNN_PATH=%%~dpF
set PATH=%CUDNN_PATH%\bin;%PATH%
```
Close the prompt and reopen it (as admin), and then reactivate the environment.

To verify that TensorFlow is working properly:
```
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
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

While CVwin is active, install the ```conda-pack``` tool:
```
conda install -y -c conda-forge conda-pack
```

I would recommend you ```cd``` to somewhere other than the default location (```C:\Windows\system32```).

Then, create your portable environment using conda-pack:
```
conda-pack --format zip
```
This will create ```CVwin.zip``` (if you opened the prompt as admin, it may be located in ```C:\Windows\system32```.)

Create a folder on your USB drive (```CVwin```) and copy ```CVwin.zip``` to this folder.  Then, unzip the files "here".

**PLEASE NOTE**: If the machine you are running this environment on does NOT have conda installed, you will NOT be able to install additional packages via conda (since the portable environment does not include the conda tool).

To activate your new environment on another machine, open a cmd prompt and type the following (replace ```K``` with the appropriate drive letter):
```
K:\CVwin\Scripts\activate.bat
```

## Portable Visual Code
Go [here](https://code.visualstudio.com/?wt.mc_id=vscom_downloads#alt-downloads) and download the zip version of Visual Code for your platform (most likely x64).
Unpack it to your USB drive.  Inside the folder for Visual Code, create a ```data``` folder:

This will cause Visual Code to store extensions and settings locally.

To run Visual Code, double-click on this version of ```Code.exe```.

Install the following extensions:
- **Python Extension Pack** by Don Jayamanne
- **Git Graph** by mhutchie

A terminal can always be created/opened with ```View menu -> Terminal```.  However, if you need to restart, click the garbage can icon on the terminal window to destroy it.

Change your default terminal from Powershell to Command Prompt:
1. ```View menu -> Command Palette -> "Terminal: Select Default Profile"```
2. Choose ```"Command Prompt"```
3. Close any existing terminals in Visual Code

Once you open a project, to make sure you are using the correct Python interpreter:
1. Close any open terminals with the garbage can icon
2. Open a .py file
3. View -> Command Palette -> "Python: Select interpreter"
4. Choose the one located at ```CVwin\python.exe``` in your USB drive
5. If the GPU is not being utilized (or you see errors about paths to CUDA libraries not being found), type the following in your terminal to manually activate your environment: ```K:\CVwin\Scripts\activate.bat``` (replace ```K``` with the appropriate drive letter for your USB drive).

## Troubleshooting

* If you need to remove the CVwin environment (the LOCAL version, not the portable one):
```
conda remove --name CVwin --all
```

* If you encounter path issues (where conda isn't found once you activate an environment), open an Anaconda prompt as admin and try the following to add conda to your path globally: 
```
conda init
```

* If Intellisense is highlighting OpenCV (cv2) commands red:
    1. Open ```File menu -> Preferences -> Settings```
    2. Select the ```User``` settings tab
    3. Search for ```pylint```
    4. Under ```Pylint:Args```, add an item: ```--generate-members```

