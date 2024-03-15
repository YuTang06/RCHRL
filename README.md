# RCHRL
This is a PyTorch implementation for our paper "Hierarchical Reinforcement Learning from Imperfect Demonstrations through Reachable Coverage-based Subgoal Filtering"

# Dependencies
* Python 3.6
* PyTorch 1.3
* OpenAI Gym
* MuJoCo
* Metaworld
 
The installation of Metaworld can be referred to [Metaworld](https://github.com/Farama-Foundation/Metaworld.git)

# Training

* AntMaze

      python main.py --env_name AntMaze --max_timesteps 2e6
  
* AntMazeSparse

      python main.py --env_name AntMazeSparse --max_timesteps 5e6

* drawer-open-v2

      python main.py --env_name drawer-open-v2 --max_timesteps 2e6

* door-open-v2
  
      python main.py --env_name door-open-v2 --max_timesteps 2e6

* door-close-v2

      python main.py --env_name door-close-v2 --max_timesteps 1e6

* reach-v2

      python main.py --env_name reach-v2 --max_timesteps 1e6

# Dataset

The demonstration data used can be found in the folder `demonstrations/`, all the experiments use our own collected datasets. 

## Regular demonstrations

In the comparative experiments, we use a well-trained HRAC hierarchical framework to collect trajectories with 100% goal arrival success rate as a regular demonstration. Regular demonstrations include: 
- `demonstrations/AntMaze_demon_subgoal_0%.csv`
- `demonstrations/AntMazeSparse_demon_subgoal_0%.csv`
- `demonstrations/drawer-open-v2_demon_subgoal_0%.csv`
- `demonstrations/door-open-v2_demon_subgoal_0%.csv`
- `demonstrations/door-close-v2_demon_subgoal_0%.csv`
- `demonstrations/reach-v2_demon_subgoal_0%.csv`

## Volatile demonstrations

In the ablation study, we used the trained HRAC framework to sample poorer trajectories with 60%-80% goal arrival success rates, and generated volatile demonstrations by mixing a portion of poorer trajectories, with a view to enabling them to deliver more chaotic noise. Volatile demonstrations include:
- `demonstrations/AntMaze_demon_subgoal_7.5%.csv`
- `demonstrations/AntMaze_demon_subgoal_10%.csv`

