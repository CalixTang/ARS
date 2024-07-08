'''
Parallel implementation of the Augmented Random Search method.
Horia Mania --- hmania@berkeley.edu
Aurelia Guy
Benjamin Recht 
'''

import parser
import time
import os
import numpy as np
# from mjrl.utils.gym_env import GymEnv
from Controller import *
from Observables import *
# from mjrl.KODex_utils.Observables import *
import logz
import ray
import utils
import optimizers
from policies import *
import socket
from shared_noise import *

from tqdm import tqdm
# import mjrl.envs
# import mj_envs   # read the env files (task files)
import time 
import graph_results
# import gym
import gymnasium as gym #for mujoco v4 and robosuite envs
from koopmanutils.env_utils import handle_extra_params, get_state_pos_and_vel_idx, instantiate_gym_env


@ray.remote
class Worker(object):
    """ 
    Object class for parallel rollout generation.
    """

    def __init__(self, env_seed,
                 task_id='',
                 policy_params = None,
                 deltas=None,
                 rollout_length=500,
                 delta_std=0.02):

        # initialize OpenAI environment for each worker
        self.env = instantiate_gym_env(task_id, policy_params)

        # each worker gets access to the shared noise table
        # with independent random streams for sampling
        # from the shared noise table. 
        self.deltas = SharedNoiseTable(deltas, env_seed + 7)
        self.policy_params = policy_params

        # initialize policy 
        self.policy = get_policy(policy_params['type'], policy_params)
        
        if policy_params['policy_checkpoint_path']:
            try:
                self.policy.update_weights(np.load(policy_params['policy_checkpoint_path'], allow_pickle = True))
            except Exception as e:
                print('Policy checkpoint path invalid')
        if policy_params['filter_checkpoint_path']:
            try:
                self.policy.observation_filter = self.policy.observation_filter.from_dict(np.load(policy_params['policy_checkpoint_path'], allow_pickle = True)[()])
            except Exception as e:
                print('Policy checkpoint path invalid')
        
            
        self.delta_std = delta_std
        self.rollout_length = rollout_length
        print("Worker initialized")
        
    def get_weights_plus_stats(self):
        """ 
        Get current policy weights and current statistics of past states.
        """
        # assert self.policy_params['type'] == 'linear'
        return self.policy.get_weights_plus_stats()
    

    def rollout(self, shift = 0., rollout_length = None):
        """ 
        Performs one rollout of maximum length rollout_length. 
        At each time-step it substracts shift from the reward.
        """
        
        if rollout_length is None:
            rollout_length = self.rollout_length

        total_reward = 0.
        steps = 0

        ob, _ = self.env.reset() #in v4 envs, there's an extra second return
        for i in range(rollout_length):
            #generate action
            action = self.policy.act(ob)
            
            reward = 0

            #(strict koopman trajectory that we follow vs doing a simple "koopman-ish" update on observed state as is implemented here)
            ob, reward, terminated, done, info = self.env.step(action) #extra 5th return in v4 gym mujoco envs...
            
            # ob, reward, done, _ = self.env.step(action)
            steps += 1
            total_reward += (reward - shift)
            if terminated or done:
                break
            
        return total_reward, steps

    def do_rollouts(self, w_policy, num_rollouts = 1, shift = 1, evaluate = False):
        """ 
        Generate multiple rollouts with a policy parametrized by w_policy.
        """

        rollout_rewards, deltas_idx = [], []
        steps = 0

        for i in range(num_rollouts):

            if evaluate:
                self.policy.update_weights(w_policy)
                deltas_idx.append(-1)
                
                # set to false so that evaluation rollouts are not used for updating state statistics
                self.policy.update_filter = False

                # for evaluation we do not shift the rewards (shift = 0) and we use the
                # default rollout length (1000 for the MuJoCo locomotion tasks)
                reward, r_steps = self.rollout(shift = 0., rollout_length = self.rollout_length)
                rollout_rewards.append(reward)
                
            else:
                idx, delta = self.deltas.get_delta(w_policy.size)
             
                delta = (self.delta_std * delta).reshape(w_policy.shape)
                deltas_idx.append(idx)

                # set to true so that state statistics are updated 
                self.policy.update_filter = True

                # compute reward and number of timesteps used for positive perturbation rollout
                self.policy.update_weights(w_policy + delta)
                pos_reward, pos_steps  = self.rollout(shift = shift)

                # compute reward and number of timesteps used for negative pertubation rollout
                self.policy.update_weights(w_policy - delta)
                neg_reward, neg_steps = self.rollout(shift = shift) 
                steps += pos_steps + neg_steps

                rollout_rewards.append([pos_reward, neg_reward])
                            
        return {'deltas_idx': deltas_idx, 'rollout_rewards': rollout_rewards, "steps" : steps}
    
    def stats_increment(self):
        self.policy.observation_filter.stats_increment()
        return

    def get_weights(self):
        return self.policy.get_weights()
    
    def get_filter(self):
        return self.policy.observation_filter

    def sync_filter(self, other):
        self.policy.observation_filter.sync(other)
        return

    
class ARSLearner(object):
    """ 
    Object class implementing the ARS algorithm.
    """

    def __init__(self, task_id='relocate',
                 policy_params=None,
                 num_workers=32, 
                 num_deltas=320, 
                 deltas_used=320,
                 delta_std=0.02, 
                 logdir=None, 
                 rollout_length=100,
                 step_size=0.01,
                 shift='constant zero',
                 params=None,
                 seed=123):
	
        print("Initialize ARSLearner object")
        logz.configure_output_dir(logdir)
        logz.save_params(params)

        # env = None
        
        #why do we need to make this every time?
        # env = gym.make(task_id)
        
        self.timesteps = 0
        # self.action_size = env.action_space.shape[0]
        # self.ob_size = env.observation_space.shape[0]
        self.num_deltas = num_deltas
        self.deltas_used = deltas_used
        self.rollout_length = rollout_length
        self.step_size = step_size
        self.delta_std = delta_std
        self.logdir = logdir
        self.shift = shift
        self.params = params
        self.max_past_avg_reward = float('-inf')
        self.num_episodes_used = float('inf')
        self.filter_type = policy_params['ob_filter']

        self.reward_threshold = params['reward_threshold']

        
        # create shared table for storing noise
        print("Creating deltas table.")
        deltas_id = create_shared_noise.remote()
        self.deltas = SharedNoiseTable(ray.get(deltas_id), seed = seed + 3)
        print('Created deltas table.')

        # initialize workers with different random seeds
        print('Initializing workers.') 
        self.num_workers = num_workers
        self.workers = [Worker.remote(seed + 7 * i,
                                      task_id = task_id,
                                      policy_params=policy_params,
                                      deltas=deltas_id,
                                      rollout_length=rollout_length,
                                      delta_std=delta_std) for i in range(num_workers)]
        print("Initialized workers.")

        # initialize policy 
        self.policy = get_policy(policy_params['type'], policy_params)
        self.w_policy = self.policy.get_weights()
            
        # initialize optimization algorithm
        self.optimizer = optimizers.SGD(self.w_policy, self.step_size)        
        print("Initialization of ARS complete.")

    def aggregate_rollouts(self, num_rollouts = None, evaluate = False):
        """ 
        Aggregate update step from rollouts generated in parallel.
        """

        if num_rollouts is None:
            num_deltas = self.num_deltas
        else:
            num_deltas = num_rollouts
            
        # put policy weights in the object store
        policy_id = ray.put(self.w_policy)

        t1 = time.time()
        num_rollouts = int(num_deltas / self.num_workers)
            
        # parallel generation of rollouts
        rollout_ids_one = [worker.do_rollouts.remote(policy_id,
                                                 num_rollouts = num_rollouts,
                                                 shift = self.shift,
                                                 evaluate=evaluate) for worker in self.workers]

        rollout_ids_two = [worker.do_rollouts.remote(policy_id,
                                                 num_rollouts = 1,
                                                 shift = self.shift,
                                                 evaluate=evaluate) for worker in self.workers[:(num_deltas % self.num_workers)]]

        # gather results 
        results_one = ray.get(rollout_ids_one)
        results_two = ray.get(rollout_ids_two)

        rollout_rewards, deltas_idx = [], [] 

        for result in results_one:
            if not evaluate:
                self.timesteps += result["steps"]
            deltas_idx += result['deltas_idx']
            rollout_rewards += result['rollout_rewards']

        for result in results_two:
            if not evaluate:
                self.timesteps += result["steps"]
            deltas_idx += result['deltas_idx']
            rollout_rewards += result['rollout_rewards']

        deltas_idx = np.array(deltas_idx)
        rollout_rewards = np.array(rollout_rewards, dtype = np.float64)
        
        print('Mean reward of collected rollouts:', rollout_rewards.mean())
        print('Maximum reward of collected rollouts:', rollout_rewards.max())
        t2 = time.time()

        print('Time to generate rollouts:', t2 - t1)
        raw_rollout_rewards = rollout_rewards[:]

        if evaluate:
            return rollout_rewards

        # select top performing directions if deltas_used < num_deltas
        max_rewards = np.max(rollout_rewards, axis = 1)
        if self.deltas_used > self.num_deltas:
            self.deltas_used = self.num_deltas
            
        idx = np.arange(max_rewards.size)[max_rewards >= np.percentile(max_rewards, 100*(1 - (self.deltas_used / self.num_deltas)))]
        deltas_idx = deltas_idx[idx]
        rollout_rewards = rollout_rewards[idx,:]
        
        
        # normalize rewards by their standard deviation - I suspect this might be leading to a div by 0...
        if np.std(rollout_rewards) > 1e-12:
            rollout_rewards /= np.std(rollout_rewards)
        else:
            print(f"Warning: Rollout reward std has gone below 1e-12: {np.std(rollout_rewards)}")
            rollout_rewards /= 1e-8

        t1 = time.time()
        # aggregate rollouts to form g_hat, the gradient used to compute SGD step
        g_hat, count = utils.batched_weighted_sum(rollout_rewards[:,0] - rollout_rewards[:,1],
                                                  (self.deltas.get(idx, self.w_policy.size)
                                                   for idx in deltas_idx),
                                                  batch_size = 500)
        # print(deltas_idx.size, flush = True)
        if deltas_idx.size != 0:
            g_hat /= deltas_idx.size
        else:
            print("Warning: Deltas_idx is of size 0.", rollout_rewards, flush = True)
        t2 = time.time()
        print('time to aggregate rollouts', t2 - t1)
        return g_hat, raw_rollout_rewards
        

    def train_step(self):
        """ 
        Perform one update step of the policy weights.
        """
        
        g_hat, rewards = self.aggregate_rollouts()                  
        print("Euclidean norm of update step:", np.linalg.norm(g_hat))
        self.w_policy -= self.optimizer._compute_step(g_hat).reshape(self.w_policy.shape)
        return rewards

    def train(self, num_iter, num_eval_rollouts = 100):
        # print("Starting training")
        training_rewards = np.zeros((num_iter, self.num_deltas * 2))
        eval_rewards = np.zeros((num_iter // 10, num_eval_rollouts))

        best_eval_policy_weights = self.w_policy #initial weights
        best_eval_policy_reward = float('-inf')
        best_filter_mean, best_filter_std = 0, 1

        start = time.time()
        for i in tqdm(range(num_iter)):
            t1 = time.time()
            step_rewards = self.train_step()
            t2 = time.time()

            #mean_reward = step_rewards.mean()
            training_rewards[i, :] = step_rewards.flatten()

            # print('total time of one step', t2 - t1)           
            # print('iter ', i,' done')

            # record statistics every 10 iterations
            if ((i + 1) % 10 == 0):
                rewards = self.aggregate_rollouts(num_rollouts = num_eval_rollouts, evaluate = True)
                w, mu, std = ray.get(self.workers[0].get_weights_plus_stats.remote())
                # np.save(self.logdir + "/koopman_policy.npy", w)

                eval_rewards[i // 10, :] = rewards.flatten()
                if rewards.mean() > best_eval_policy_reward:
                    best_eval_policy_reward = rewards.mean()
                    best_eval_policy_weights = w
                    np.save(os.path.join(self.logdir, 'best_koopman_policy_weights.npy'), w)

                    if self.filter_type == 'MeanStdFilter':
                        best_filter_mean, best_filter_std = mu, std
                        ob_filter_obj = self.policy.get_observation_filter().as_dict()
                        np.save(os.path.join(self.logdir, 'best_obs_filter.npy'), ob_filter_obj)

                # save latest policy weights and latest filter
                np.save(os.path.join(self.logdir, 'latest_koopman_policy_weights.npy'), w)        

                if self.filter_type == 'MeanStdFilter':
                    ob_filter_obj = self.policy.get_observation_filter().as_dict()
                    np.save(os.path.join(self.logdir, 'latest_obs_filter.npy'), ob_filter_obj)
                
                #eval logging
                # print(sorted(self.params.items()))
                logz.log_tabular("Time", time.time() - start)
                logz.log_tabular("Iteration", i + 1)
                logz.log_tabular("AverageReward", np.mean(rewards))
                logz.log_tabular("StdRewards", np.std(rewards))
                logz.log_tabular("MaxRewardRollout", np.max(rewards))
                logz.log_tabular("MinRewardRollout", np.min(rewards))
                logz.log_tabular("timesteps", self.timesteps)
                logz.dump_tabular()
                
            t1 = time.time()
            # get statistics from all workers
            for j in range(self.num_workers):
                self.policy.observation_filter.update(ray.get(self.workers[j].get_filter.remote()))
            self.policy.observation_filter.stats_increment()

            # make sure master filter buffer is clear
            self.policy.observation_filter.clear_buffer()
            # sync all workers
            filter_id = ray.put(self.policy.observation_filter)
            setting_filters_ids = [worker.sync_filter.remote(filter_id) for worker in self.workers]
            # waiting for sync of all workers
            ray.get(setting_filters_ids)
         
            increment_filters_ids = [worker.stats_increment.remote() for worker in self.workers]
            # waiting for increment of all workers
            ray.get(increment_filters_ids)            
            t2 = time.time()
            print('Time to sync statistics:', t2 - t1, flush = True)

            if ((i + 1) % 10 == 0):
                if best_eval_policy_reward > self.reward_threshold:
                    print(f'Eval reward threshold of {self.reward_threshold} reached before training time limit. Ending training early at iteration {i + 1}.')
                    break
        
        #save best weights
        print(f'Best eval policy mean reward: {best_eval_policy_reward}')
        np.save(os.path.join(self.logdir, 'best_koopman_policy_weights.npy'), best_eval_policy_weights)
        np.save(os.path.join(self.logdir, 'latest_koopman_policy_weights.npy'), self.w_policy)        
 
        np.save(os.path.join(self.logdir, 'training_rewards.npy'), training_rewards)
        np.save(os.path.join(self.logdir, 'eval_rewards.npy'), eval_rewards)


        if self.filter_type == 'MeanStdFilter':
            ob_filter_obj = self.policy.get_observation_filter().as_dict()
            np.save(os.path.join(self.logdir, 'obs_filter.npy'), ob_filter_obj)

            best_ob_filter_obj = self.policy.get_observation_filter().copy()
            best_ob_filter_obj.mean, best_ob_filter_obj.std = best_filter_mean, best_filter_std
            np.save(os.path.join(self.logdir, 'best_filter.npy'), ob_filter_obj)



        return training_rewards, eval_rewards


def run_ars(params):

    dir_path = params['dir_path']
    logdir = None

    if not(os.path.exists(dir_path)):
        os.makedirs(dir_path)
    if params.get('run_name', None) is not None:
        logdir = os.path.join(dir_path, params['run_name'])
    else:
        logdir = os.path.join(dir_path, str(time.time_ns()))
        while os.path.exists(logdir):
            logdir = os.path.join(dir_path, str(time.time_ns()))
    print(f"Logging to directory {logdir}")
    os.makedirs(logdir)

    if params.get('reward_threshold', None) is None or params.get('reward_threshold', None) < 0:
        params['reward_threshold'] = float('inf')

    #surely there's a better way to get the ob and ac dims
    env = gym.make(params['task_id'])

    ob_dim = 0
    if isinstance(env.observation_space, gym.spaces.dict.Dict):
        ob_dim = sum([v.shape[0] for k, v in env.observation_space.items()])
    else:
        ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]
    print(ob_dim, ac_dim)
    # ob_dim, ac_dim = params['robot_dim'] + params['obj_dim'], 0

    PID_P = 0.1
    PID_D = 0.001 
    pd_controller = PID(PID_P, 0.0, PID_D)

    state_pos_idx, state_vel_idx = get_state_pos_and_vel_idx(params['task_id'])

    # set policy parameters. Possible filters: 'MeanStdFilter' for v2, 'NoFilter' for v1.
    policy_params={'type':params['policy_type'],
                   'ob_filter':params['filter'],
                   'ob_dim':ob_dim,
                   'ac_dim':ac_dim, 
                   'num_modes': params['num_modes'], # only for EigenRelocate policy
                   'PID_controller': pd_controller,
                   'policy_checkpoint_path': params.get('policy_checkpoint_path', None),
                   'filter_checkpoint_path': params.get('filter_checkpoint_path', None),
                   'lifting_function': params.get('lifting_function', 'locomotion'),
                   'obs_pos_idx': state_pos_idx,
                   'obs_vel_idx': state_vel_idx,
                   'seed': params['seed']
                   }
    
    handle_extra_params(params, policy_params)

    print(f"ARS parameters: {params}")
    print(f"Policy parameters: {policy_params}", flush = True)
    ARS = ARSLearner(task_id=params['task_id'],
                     policy_params=policy_params,
                     num_workers=params['n_workers'], 
                     num_deltas=params['n_directions'],
                     deltas_used=params['deltas_used'],
                     step_size=params['step_size'],
                     delta_std=params['delta_std'], 
                     logdir=logdir,
                     rollout_length=params['rollout_length'],
                     shift=params['shift'],
                     params=params,
                     seed = params['seed'])
        
    train_rewards, eval_rewards = ARS.train(params['n_iter'])
    graph_results.graph_training_and_eval_rewards(train_rewards, eval_rewards, logdir, False)
    print(f"All files written to {logdir}", flush = True)

    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    #ARS arguments
    
    # Mujoco envs: Swimmer, Hopper, Walker2d, HalfCheetah, Ant, Humanoid (all -v4)
    # FrankaKitchen: FrankaKitchen-v1
    # Shadow Hand: HandReach, HandManipulateBlock, HandManipulateEgg, HandManipulatePen (all -v1). add _BooleanTouchSensor or _ContinuousTouchSensor before version for all but HandReach
    parser.add_argument('--task_id', type=str, default='Hopper-v4')
    parser.add_argument('--n_iter', '-n', type=int, default=4000) #training steps
    parser.add_argument('--n_directions', '-nd', type=int, default=128) #directions explored - results in 2*d actual policies
    parser.add_argument('--deltas_used', '-du', type=int, default=64) #directions kept for gradient update
    parser.add_argument('--step_size', '-s', type=float, default=0.2)#0.02, alpha in the paper #0.04
    parser.add_argument('--delta_std', '-std', type=float, default=0.25)# 0.03, v in the paper #4e-3
    parser.add_argument('--n_workers', '-e', type=int, default = 8)
    parser.add_argument('--rollout_length', '-r', type=int, default=1000) #100 timesteps * 5 b/c of the PID subsampling
    # for Swimmer-v1 and HalfCheetah-v1 use shift = 0
    # for Hopper-v1, Walker2d-v1, and Ant-v1 use shift = 1
    # for Humanoid-v1 used shift = 5
    parser.add_argument('--shift', type=float, default=1) #TODO: tweak as necessary
    parser.add_argument('--seed', type=int, default=237)
    parser.add_argument('--policy_type', type=str, default='truncatedkoopman')
    parser.add_argument('--dir_path', type=str, default='data')
    # for ARS V1 use filter = 'NoFilter', V2 = 'MeanStdFilter'
    parser.add_argument('--filter', type=str, default='NoFilter') 
    
    #Corresponds to the type of observable to use. For now, either identity, locomotion, or manipulation
    parser.add_argument('--lifting_function', type=str, default = 'locomotion')

    #eigenkoopman arg
    parser.add_argument('--num_modes', type=int, default = 0) #EigenRelocate only, for relocate task in [1, 759]
    
    #utility arguments
    parser.add_argument('--params_path', type = str)
    parser.add_argument('--policy_checkpoint_path', type = str)
    parser.add_argument('--filter_checkpoint_path', type = str)
    parser.add_argument('--run_name', type = str)

    #added to allow for running only until a certain performance threshold is reached 
    parser.add_argument('--reward_threshold', type = float, default = float('inf')) 
    
   
    
    
    args = parser.parse_args()
    params = vars(args)

    ray.init(num_cpus=args.n_workers)

    if args.params_path is not None:
        import json
        params = json.load(open(args.params_path, 'r'))
        # print(params)
        if args.run_name:
            params['run_name'] = args.run_name
    
    run_ars(params)
