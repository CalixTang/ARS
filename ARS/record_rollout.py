import parser
import os
import numpy as np
import gym
from policies import *
from shared_noise import *
from Observables import *
from Controller import *
import numpy as np
from tqdm import tqdm
import imageio
import os
from filter import Filter

#modified version of run_policy.py and record_relocate from mjrl.utils.gym_env
def record_rollouts(task_id='HalfCheetah-v2',
                policy_params = None,
                policy = None,
                logdir=None,
                num_rollouts=50,
                rollout_length = 500,
                shift = 0.,
                vid_res = [720, 640],
                seed=123):
        
    env = None 

    env = gym.make(task_id, render_mode = 'rgb_array')
    # env.seed(seed)
    env.reset(seed = seed) #for v4 envs

    save_path = os.path.join(logdir, f"{task_id}_eval_{num_rollouts}_rollouts.mp4")
    vid_writer = imageio.get_writer(save_path, mode = 'I', fps = 60)
    
    # env.viewer_setup()
    
    total_reward = 0.
    steps = 0

    ep_rewards = np.zeros(num_rollouts)

    for i in tqdm(range(num_rollouts)):

        ob, _ = env.reset() #for v4 envs
        episode_reward = 0
        for t in range(rollout_length):
            #generate torque action
            action = policy.act(ob)
            reward = 0

            #TODO: verify if we need to be using koopman op on the next_o or the actually observed env state
            #(strict koopman trajectory that we follow vs doing a simple "koopman-ish" update on observed state as is implemented here)
            ob, reward, terminated, done, info = env.step(action)   #for v4 env

            # res = env.render(mode = 'rgb_array')
            res = env.render() 
            # res = env.render(mode = 'rgb_array', width = vid_res[0], height = vid_res[1])
            vid_writer.append_data(res)

            episode_reward += (reward - shift)
            if terminated or done:
                break
        
        ep_rewards[i] = episode_reward
                
    vid_writer.close()
    return ep_rewards

if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser()

    parser.add_argument('--logdir', type = str, required = True, default = 'data')
    parser.add_argument('--policy_weight_file', type=str, default = 'best_koopman_policy_weights.npy') #this should be contained in logdir
    parser.add_argument('--filter_file', type=str, default = 'best_filter.npy') #this should be contained in logdir
    parser.add_argument('--num_rollouts', type = int, default = 20)

    args = parser.parse_args()

    params = json.load(open(os.path.join(args.logdir, 'params.json'), 'r'))
    policy_path = os.path.join(args.logdir, args.policy_weight_file)

    env = gym.make(params['task_id'])

    ob_dim, ac_dim = env.observation_space.shape[0], env.action_space.shape[0]
    PID_P = 0.1
    PID_D = 0.001  
    Simple_PID = PID(PID_P, 0.0, PID_D)

    state_pos_idx, state_vel_idx = get_state_pos_and_vel_idx(params['task_id'])


    # set policy parameters. Possible filters: 'MeanStdFilter' for v2, 'NoFilter' for v1.
    policy_params={'type':params['policy_type'],
                   'ob_filter':params['filter'],
                   'ob_dim':ob_dim,
                   'ac_dim':ac_dim, 
                #    'robot_dim': params['robot_dim'],
                #    'obj_dim': params['obj_dim'],
                #    'object': params['object'],
                # 'num_modes': params['num_modes'],#only for EigenRelocate
                   'PID_controller': Simple_PID,
                   'lifting_function': params.get('lifting_function', 'locomotion'),
                   'obs_pos_idx': state_pos_idx,
                   'obs_vel_idx': state_vel_idx
                   }
    
    policy = get_policy(policy_params['type'], policy_params)
    

    policy_w = np.load(policy_path, allow_pickle = True)
    policy.update_weights(policy_w)
    policy.update_filter = False

    if policy_params['ob_filter'] != 'NoFilter':
        policy.observation_filter = policy.observation_filter.from_dict(np.load(os.path.join(args.logdir, args.filter_file), allow_pickle = True)[()])
        # print(x)
    

    record_rollouts(task_id=params['task_id'],
                policy_params = policy_params,
                policy = policy,
                logdir = args.logdir,
                num_rollouts=args.num_rollouts,
                rollout_length = params['rollout_length'],
                shift = params['shift'],
                vid_res = [720, 640],
                seed=params['seed'])