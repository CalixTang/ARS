import parser
import os
import numpy as np
# import gym
from mjrl.utils.gym_env import GymEnv
from policies import *
from shared_noise import *
from mjrl.KODex_utils.Observables import *
from mjrl.KODex_utils.Controller import *
import mjrl.envs
import mj_envs   # read the env files (task files)
import numpy as np
from tqdm import tqdm
import imageio
import os
from filter import Filter

# modified version of run_policy.py and record_relocate from mjrl.utils.gym_env
def record_rollouts(task_id='relocate',
                policy_params = None,
                policy = None,
                logdir=None,
                num_rollouts=50,
                rollout_length = 500,
                shift = 0.,
                vid_res = [720, 640],
                seed=123):
        
    env = None 

    if task_id == 'relocate':
        env = GymEnv('relocate-v0', "PID", policy_params['object'])
    else:
        env = GymEnv(task_id, "PID")
    env.seed(seed)

    save_path = os.path.join(logdir, f"{task_id}_eval_{num_rollouts}_rollouts.mp4")
    vid_writer = imageio.get_writer(save_path, mode = 'I', fps = 60)
    
    env.env.mj_viewer_setup()
    
    total_reward = 0.
    steps = 0

    ep_rewards = np.zeros(num_rollouts)

    for i in tqdm(range(num_rollouts)):

        _ = env.reset()
        episode_reward = 0

        for t in range(rollout_length):
            #observe full state b/c we need it
            ob = env.get_env_state()
            action = policy.get(ob)

            #(strict koopman trajectory that we follow vs doing a simple "koopman-ish" update on observed state as is implemented here)
            ob, reward, done, info = env.step(action) 

            res = env.env.viewer._read_pixels_as_in_window(resolution = vid_res)
            vid_writer.append_data(res) 
            
            # ob, reward, done, _ = self.env.step(action)
            steps += 1
            total_reward += (reward - shift)
            if done:
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

    ob_dim, ac_dim = params['robot_dim'] + params['obj_dim'], 0
    PID_P = 10
    PID_D = 0.005  
    Simple_PID = PID(PID_P, 0.0, PID_D)

    if not params['object'] or params['object'] == 'ball':
        params['object'] = ''

    # set policy parameters. Possible filters: 'MeanStdFilter' for v2, 'NoFilter' for v1.
    policy_params={'type':params['policy_type'],
                   'ob_filter':params['filter'],
                   'ob_dim':ob_dim,
                   'ac_dim':ac_dim, 
                   'robot_dim': params['robot_dim'],
                   'obj_dim': params['obj_dim'],
                   'object': params['object'],
                   'PID_controller': Simple_PID,
                   'num_modes': params['num_modes']} #only for EigenRelocate
    
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