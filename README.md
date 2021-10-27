# Traffic Light Control Baselines

This repo provides OpenAI Gym compatible environments for traffic light control scenario and a bunch of baseline methods. 

Environments include single intersetion (single-agent) and multi intersections (multi-agents) with different road network and traffic flow settings.

Baselines include traditional Taffic Signal Control algorithms and reinforcement learning based methods.


## Installation

```
conda env create -f requirement.yml

conda activate LibSignal

python run_dqn.py cityflow.cfg
```
