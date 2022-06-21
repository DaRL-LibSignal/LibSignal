# Traffic Light Control Baselines

This repo provides OpenAI Gym compatible environments for traffic light control scenario and a bunch of baseline methods. 

Environments include single intersetion (single-agent) and multi intersections (multi-agents) with different road network and traffic flow settings.

Baselines include traditional Taffic Signal Control algorithms and reinforcement learning based methods.

## Introduction

LibSignal is a cross-simulator environment that provides multiple traditional and Reinforcement Learning models in traffic control tasks. Currently, we support SUMO, CityFlow, and CBEine simulation environments. Conversion between SUMO and CityFlow is carefully calibrated.

## Installation

```
conda env create -f requirement.yml

conda activate LibSignal

python run.py --agent dqn --cityflow_path configs/cityflow4X4.cfg
```
