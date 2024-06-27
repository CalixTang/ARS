# Code in this file is copied and adapted from
# https://github.com/openai/evolution-strategies-starter.

import numpy as np
import gymnasium as gym

def instantiate_gym_env(task_id, policy_params):
    task_name = task_id.split('-')[0]

    if task_name == 'FrankaKitchen':
        pass
    elif 'Fetch' in task_name:
        env = gym.make(task_id, max_episode_steps = policy_params['rollout_length'], reward_type = policy_params['reward_type'], render_mode = policy_params.get('render_mode', None), width = policy_params.get('vid_res', [0])[0], height = policy_params.get('vid_res', [0, 0])[1])
    elif 'HandManipulate' in task_name:
        env = gym.make(task_id, max_episode_steps = policy_params['rollout_length'], reward_type = policy_params['reward_type'], render_mode = policy_params.get('render_mode', None), width = policy_params.get('vid_res', [0])[0], height = policy_params.get('vid_res', [0, 0])[1])
    else:
        env = gym.make(task_id, render_mode = policy_params.get('render_mode', None), width = policy_params.get('vid_res', [0])[0], height = policy_params.get('vid_res', [0, 0])[1])
        
    env.reset(seed = policy_params['seed'])

    return env

def handle_extra_params(params, policy_params):
    task_name = params['task_id'].split('-')[0]
    if task_name == 'FrankaKitchen':
        pass
    elif 'HandManipulate' in task_name:
        policy_params['rollout_length'] = params.get('rollout_length', 50)
        policy_params['reward_type'] = params.get('reward_type', 'dense') #dense or sparse
    elif 'Fetch' in task_name:
        policy_params['rollout_length'] = params.get('rollout_length', 50)
        policy_params['reward_type'] = params.get('reward_type', 'dense') #dense or sparse

    return


def itergroups(items, group_size):
    assert group_size >= 1
    group = []
    for x in items:
        group.append(x)
        if len(group) == group_size:
            yield tuple(group)
            del group[:]
    if group:
        yield tuple(group)



def batched_weighted_sum(weights, vecs, batch_size):
    total = 0
    num_items_summed = 0
    for batch_weights, batch_vecs in zip(itergroups(weights, batch_size),
                                         itergroups(vecs, batch_size)):
        assert len(batch_weights) == len(batch_vecs) <= batch_size
        total += np.dot(np.asarray(batch_weights, dtype=np.float64),
                        np.asarray(batch_vecs, dtype=np.float64))
        num_items_summed += len(batch_weights)
    return total, num_items_summed
