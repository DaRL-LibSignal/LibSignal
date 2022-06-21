# Introduction
This repo provides OpenAI Gym compatible environments for traffic light control scenario and a bunch of baseline methods. 

Environments include single intersetion (single-agent) and multi intersections (multi-agents) with different road network and traffic flow settings.

Baselines include traditional Taffic Signal Control algorithms and reinforcement learning based methods.

LibSignal is a cross-simulator environment that provides multiple traditional and Reinforcement Learning models in traffic control tasks. Currently, we support SUMO, CityFlow, and CBEine simulation environments. Conversion between SUMO and CityFlow is carefully calibrated.

# Install

## Source

LibSingal provides installation from source code.
Please execute the following command to install and configure  our environment.

```
mkdir DaRL
cd DaRL
git clone git@github.com:DaRL-LibSignal/LibSignal.git
```

## Simulator environment configuration
<br />
Though CityFlow and SUMO are stable under Windows and Linux systems, we still recommend users work under the Linux system. Currently, CBEngine is stable under the Linux system.<br><br>

### CityFlow Environment
<br />

To install CityFlow simulator, please follow the instruction on [CityFlow Doc](https://cityflow.readthedocs.io/en/latest/install.html#)


```
sudo apt update && sudo apt install -y build-essential cmake

git clone https://github.com/cityflow-project/CityFlow.git
cd CityFlow
pip install .
```
To test configuration:
```
import cityflow
env = cityflow.Engine
```
<br>

### SUMO Environment
<br />

To install SUMO environment, please follow the instruction on [SUMO Doc](https://epics-sumo.sourceforge.io/sumo-install.html#)

```
sudo apt-get install cmake python g++ libxerces-c-dev libfox-1.6-dev libgdal-dev libproj-dev libgl2ps-dev swig

git clone --recursive https://github.com/eclipse/sumo

export SUMO_HOME="$PWD/sumo"
mkdir sumo/build/cmake-build && cd sumo/build/cmake-build
cmake ../..
make -j$(nproc)
```
To test installation:
```
cd ~/DaRL/sumo/bin
./sumo
```

To add SUMO and traci model into system PATH, execute code below:
```
export SUMO_HOME=~/DaRL/sumo
export PYTHONPATH="$SUMO_HOME/tools:$PYTHONPATH"
```
To test configuration:
```
import libsumo
import traci
```
<br>

### CBEngine
<br />

CBEngine currently works stably under the Linux system; we highly recommend users choose Linux if we plan to conduct experiments under the CBEinge simulation environment. (Currently not available)

<br>


## Requirment
<br />

Our code is based on Python version 3.9 and Pytorch version 1.11.0. For example, if your CUDA version is 11.3 you can follow the instruction on [PyTorch](https://pytorch.org/get-started/locally/)

```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

pip install -r requirements.txt
```

## Selective agents
<br />

We also support agents implemented based on other libraries
```
# Colight Geometric implementation based on default environment mentioned in Requirment

pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.11.0+cu113.html

# ppo_pfrl implementation
pip install pfrl
```
Detailed instrcuctions can be found on page [Pytorch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) and [PFRL](https://pfrl.readthedocs.io/en/latest/install.html). After installation, user should uncomment code in PATH ./agent/\_\_init\_\_.py 
```
# from .ppo_pfrl import IPPO_pfrl
# from colight import CoLightAgent
```
