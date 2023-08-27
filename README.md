# Introduction
This repo provides OpenAI Gym-compatible environments for traffic light control scenarios and a bunch of baseline methods. 

Environments include a single intersection (single-agent) and multi-intersections (multi-agents) with different road networks and traffic flow settings.

Baselines include traditional Traffic Signal Control algorithms and reinforcement learning-based methods.

LibSignal is a cross-simulator environment that provides multiple traditional and Reinforcement Learning models in traffic control tasks. Currently, we support SUMO, CityFlow, and CBEine simulation environments. Conversion between SUMO and CityFlow is carefully calibrated.

# Install

## Docker image
We configurated a ready-to-use Docker image of the LibSignal environment and it can be used as the interface to interact directly with the server side. 

### How to use docker





## Source

LibSingal provides installation from the source code.
Please execute the following command to install and configure our environment.


### Download our library from GitHub

```
mkdir DaRL
cd DaRL
git clone git@github.com:DaRL-LibSignal/LibSignal.git
```

### Simulator environment configuration
<br />
Though CityFlow and SUMO are stable under Windows and Linux systems, we still recommend users work under the Linux system. Currently, CBEngine is stable under the Linux system.<br><br>

### CityFlow Environment
<br />

To install the CityFlow simulator, please follow the instructions on [CityFlow Doc](https://cityflow.readthedocs.io/en/latest/install.html#)


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

To install the SUMO environment, please follow the instructions on [SUMO Doc](https://epics-sumo.sourceforge.io/sumo-install.html#)

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

### Converter
<br />

We provide a converter to transform configurations including road net and traffic flow files across CityFlow and SUMO. More details in [converter.py](./common/converter.py)

To convert from CityFlow to SUMO: 

```

python converter.py --typ c2s --or_cityflownet CityFlowNetPath --sumonet ConvertedSUMONetPath --or_cityflowtraffic CityFlowTrafficPath --sumotraffic ConvertedSUMOTrafficPath 

```

To convert from SUMO to CityFlow: 
```
python converter.py --typ s2c --or_sumonet SUMONetPath --cityflownet ConvertedCityFlowNetPath --or_sumotraffic SUMOTrafficPath --cityflowtraffic ConvertedCityFlowTrafficPath --sumocfg SUMOConfigs
```
After running the code, the converted traffic network files, traffic flow files, and some intermediate files will be generated in the specified folder.

<br>

## Requirement
<br />

Our code is based on Python version 3.9 and Pytorch version 1.11.0. For example, if your CUDA version is 11.3 you can follow the instructions on [PyTorch](https://pytorch.org/get-started/locally/)

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
# Start

## Run Model Pipeline

Our library has a uniform structure that empowers users to start their experiments with just one click. Users can start an experiment by setting arguments in the run.py file and start with their customized settings. The following part is the arguments provided to customize.

```
python run.py
```

Supporing parameters:

- <font color=red> thread_num:  </font> number of threads for cityflow simulation

- <font color=red> ngpu:  </font> how many gpu resources used in this experiment

- <font color=red> task:  </font> task type to run

- <font color=red> agent:  </font> agent type of agents in RL environment

- <font color=red> world:  </font> simulator type

- <font color=red> dataset:  </font> type of dataset in the training process

- <font color=red> path:  </font> path to the configuration file

- <font color=red> prefix:  </font> the number of prefix in this running process

- <font color=red> seed:  </font> seed for PyTorch backend
  </br></br>


# Maintaining plan

*<font size=4>To ensure the stability of our traffic signal testbed, we will first push new code onto **dev** branch, after validation, and then merge it into the master branch. </font>*

| **UPdate index**           | **Date**      | **Status** | **Merged** |
|----------------------------|---------------|------------|------------|
| **MPLight implementation** | July-18-2022  | developed  | √          |
| **Libsumo integration**    | August-8-2022 | developed | √          |
| **Delay calculation**      | August-8-2022 | developed |  √          |
| **CoLight adaptation for heterogenous network** | September-1-2022 | developling |  |
| **Optimize FRAP and MPLight**      | October-4-2022 | developed |  √          |
| **FRAP adaptation for irregular intersections**      | October-18-2022 | developed |  √          |
| **PettingZoo envrionment to better support MARL**      | Jul-18-2023 | developed |       |
| **RLFX Agent controlling phase and duration**      | Jul-18-2023 | developed |    |
| **Ray rllib support**      | Jul-18-2023 | developling |   |

# Citation


A whitepaper of LibSignal accepted by NeurIPS 2022 Workshop: Reinforcement Learning for Real Life is available https://arxiv.org/abs/2211.10649, and can be cited with the following bibtex entry:

```
@misc{2211.10649,
  Author = {Mei, Hao and Lei, Xiaoliang Lei and Da, Longchao Da and Shi, Bin and Wei, Hua},
  Title = {LibSignal: An Open Library for Traffic Signal Control},
  Year = {2022},
  booktitle={NeurIPS Workshop on Reinforcement Learning for Real Life},
}
```
