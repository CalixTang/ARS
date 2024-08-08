# ARS for Koopman Policies
This repository is forked from the [original ARS repo](https://github.com/modestyachts/ARS) for use with the [Koopman-RL project](https://github.com/CalixTang/Koopman-RL). The rest of this README is partially updated from the original README.

## Augmented Random Search (ARS)

ARS is a random search method for training linear policies for continuous control problems, based on the paper ["Simple random search provides a competitive approach to reinforcement learning."](https://arxiv.org/abs/1803.07055) 

## Prerequisites for running ARS

Our ARS implementation relies on Python 3, ~~OpenAI Gym version 0.9.3~~ gymnasium version 0.28.1, ~~mujoco-py 0.5.7, MuJoCo Pro version 1.31,~~ and the Ray library for parallel computing.  

To install Gymnasium and the robotics suite, follow the [instructions here](https://robotics.farama.org/content/installation/) (for most purposes `pip install gymnasium-robotics` is enough). 

To install Ray execute:
``` 
pip install ray
```
For more information on Ray see http://ray.readthedocs.io/en/latest/. 

Both `gymnasium-robotics` and `ray` may be installed from the KoopmanRL project conda environment already.

## Running ARS

Assuming you are within the `ARS` project directory:
```
python ARS/ars.py
```

All arguments passed into ARS are optional and can be modified to train other environments, use different hyperparameters, or use different random seeds.

For example, to train a policy for Humanoid-v4, execute the following command:

```
python code/ars.py --env_name Humanoid-v4 --n_directions 230 --deltas_used 230 --step_size 0.02 --delta_std 0.0075 --n_workers 48 --shift 5
```

## Rendering Trained Policy

To render a trained policy, execute a command of the following form:

```
python ARS/record_rollout.py --logdir ./example_run --num_rollouts 20
```

