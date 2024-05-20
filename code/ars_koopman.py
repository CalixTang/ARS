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
# import gym
from mjrl.utils.gym_env import GymEnv
import logz
import ray
import utils
import optimizers
from policies import *
import socket
from shared_noise import *
from mjrl.KODex_utils.Observables import *
from tqdm import tqdm
import mjrl.envs
import mj_envs   # read the env files (task files)
import time 


@ray.remote
class Worker(object):
    import mj_envs
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
        if task_id == 'relocate':
            self.env = GymEnv('relocate-v0', "PID", policy_params['object'])
        else:
            self.env = GymEnv(task_id, "PID")
        self.env.seed(env_seed)

        # each worker gets access to the shared noise table
        # with independent random streams for sampling
        # from the shared noise table. 
        self.deltas = SharedNoiseTable(deltas, env_seed + 7)
        self.policy_params = policy_params
        if policy_params['type'] == 'linear':
            self.policy = LinearPolicy(policy_params)
        elif policy_params['type'] == 'relocate':
            self.policy = RelocatePolicy(policy_params)
        else:
            raise NotImplementedError
            
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

        _ = self.env.reset()
        for i in range(rollout_length):

            #observe full state b/c we need it
            ob = self.env.get_env_state()

            #generate torque action
            action = self.policy.act(ob)
            
            reward = 0

            #TODO: verify if we need to be using koopman op on the next_o or the actually observed env state
            #(strict koopman trajectory that we follow vs doing a simple "koopman-ish" update on observed state as is implemented here)
            next_o, reward, done, goal_achieved = self.env.step(action)  
            
            # ob, reward, done, _ = self.env.step(action)
            steps += 1
            total_reward += (reward - shift)
            if done:
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

        env = None
        
        if task_id == 'relocate':
            if not policy_params['object'] or policy_params['object'] == 'ball':
                policy_params['object'] = ''
            env = GymEnv('relocate-v0', "PID", policy_params['object'])
        else:
            env = GymEnv(task_id, "PID")
        
        self.timesteps = 0
        self.action_size = env.action_space.shape[0]
        self.ob_size = env.observation_space.shape[0]
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
        if policy_params['type'] == 'linear':
            self.policy = LinearPolicy(policy_params)
            self.w_policy = self.policy.get_weights()
        elif policy_params['type'] == 'relocate':
            self.policy = RelocatePolicy(policy_params)
            self.w_policy = self.policy.get_weights()
        else:
            raise NotImplementedError
            
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

        if evaluate:
            return rollout_rewards

        # select top performing directions if deltas_used < num_deltas
        max_rewards = np.max(rollout_rewards, axis = 1)
        if self.deltas_used > self.num_deltas:
            self.deltas_used = self.num_deltas
            
        idx = np.arange(max_rewards.size)[max_rewards >= np.percentile(max_rewards, 100*(1 - (self.deltas_used / self.num_deltas)))]
        deltas_idx = deltas_idx[idx]
        rollout_rewards = rollout_rewards[idx,:]
        
        # normalize rewards by their standard deviation
        rollout_rewards /= np.std(rollout_rewards)

        t1 = time.time()
        # aggregate rollouts to form g_hat, the gradient used to compute SGD step
        g_hat, count = utils.batched_weighted_sum(rollout_rewards[:,0] - rollout_rewards[:,1],
                                                  (self.deltas.get(idx, self.w_policy.size)
                                                   for idx in deltas_idx),
                                                  batch_size = 500)
        g_hat /= deltas_idx.size
        t2 = time.time()
        print('time to aggregate rollouts', t2 - t1)
        return g_hat, rollout_rewards
        

    def train_step(self):
        """ 
        Perform one update step of the policy weights.
        """
        
        g_hat, rewards = self.aggregate_rollouts()                  
        print("Euclidean norm of update step:", np.linalg.norm(g_hat))
        self.w_policy -= self.optimizer._compute_step(g_hat).reshape(self.w_policy.shape)
        return rewards

    def train(self, num_iter, num_eval_rollouts = 100):
        print("Starting training")
        training_rewards = np.zeros((num_iter, self.deltas_used * 2))
        eval_rewards = np.zeros((num_iter // 10, num_eval_rollouts))

        best_eval_policy_weights = self.w_policy #initial weights
        best_eval_policy_reward = float('-inf')

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
                print(f"It {i}", flush = True)
                rewards = self.aggregate_rollouts(num_rollouts = num_eval_rollouts, evaluate = True)
                w = ray.get(self.workers[0].get_weights_plus_stats.remote())
                np.save(self.logdir + "/koopman_policy.npy", w)

                eval_rewards[i // 10, :] = rewards.flatten()
                if rewards.mean() > best_eval_policy_reward:
                    best_eval_policy_reward = rewards.mean()
                    best_eval_policy_weights = w
                
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
            print('Time to sync statistics:', t2 - t1)

        #save best weights
        print(f'Best eval policy mean reward: {best_eval_policy_reward}')
        np.save(os.path.join(self.logdir, 'best_koopman_policy_weights.npy'), best_eval_policy_weights)
        np.save(os.path.join(self.logdir, 'latest_koopman_policy_weights.npy'), self.w_policy)        
 
        np.save(os.path.join(self.logdir, 'training_rewards.npy'), training_rewards)
        np.save(os.path.join(self.logdir, 'eval_rewards.npy'), eval_rewards)
        return 

def run_ars(params):

    dir_path = params['dir_path']

    if not(os.path.exists(dir_path)):
        os.makedirs(dir_path)
    logdir = os.path.join(dir_path, str(time.time_ns()))
    if not(os.path.exists(logdir)):
        print(f"Making dir {logdir}")
        os.makedirs(logdir)

    #i think we don't care about these for our case?
    # ob_dim = env.observation_space.shape[0]
    # ac_dim = env.action_space.shape[0]
    # print(ob_dim, ac_dim)
    ob_dim, ac_dim = 0, 0

    PID_P = 10
    PID_D = 0.005  
    Simple_PID = PID(PID_P, 0.0, PID_D)

    # set policy parameters. Possible filters: 'MeanStdFilter' for v2, 'NoFilter' for v1.
    policy_params={'type':params['policy_type'],
                   'ob_filter':params['filter'],
                   'ob_dim':ob_dim,
                   'ac_dim':ac_dim, 
                   'robot_dim': params['robot_dim'],
                   'obj_dim': params['obj_dim'],
                   'object': params['object'],
                   'PID_controller': Simple_PID}
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
        
    ARS.train(params['n_iter'])
   
    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=str, default='relocate')
    parser.add_argument('--n_iter', '-n', type=int, default=300) #training steps
    parser.add_argument('--n_directions', '-nd', type=int, default=16) #directions explored - results in 2*d actual policies
    parser.add_argument('--deltas_used', '-du', type=int, default=8) #directions kept for gradient update
    parser.add_argument('--step_size', '-s', type=float, default=.05)#0.02, alpha in the paper
    parser.add_argument('--delta_std', '-std', type=float, default=0.02)# 0.03, v in the paper
    parser.add_argument('--n_workers', '-e', type=int, default=1)
    parser.add_argument('--rollout_length', '-r', type=int, default=500) #100 timesteps * 5 b/c of the PID subsampling

    # for Swimmer-v1 and HalfCheetah-v1 use shift = 0
    # for Hopper-v1, Walker2d-v1, and Ant-v1 use shift = 1
    # for Humanoid-v1 used shift = 5
    parser.add_argument('--shift', type=float, default=0) #TODO: tweak as necessary
    parser.add_argument('--seed', type=int, default=237)
    parser.add_argument('--policy_type', type=str, default='relocate')
    parser.add_argument('--dir_path', type=str, default='data')

    # for ARS V1 use filter = 'NoFilter', V2 = 'MeanStdFilter'
    parser.add_argument('--filter', type=str, default='MeanStdFilter') 

    #for relocate task, allow different object
    parser.add_argument('--object', type=str, default = 'ball')
    parser.add_argument('--robot_dim', type=int, default = 30)
    parser.add_argument('--obj_dim', type=int, default = 12)
    #parser.add_argument('--env_init_path', type=str, default = 'Samples/Relocate/Relocate_task_20000_samples.pickle')
    if ray.is_initialized():
        ray.shutdown()
    
    local_ip = socket.gethostbyname(socket.gethostname())
    ray.init(address = local_ip + ':6379')
    
    args = parser.parse_args()
    params = vars(args)
    
    trained_policy = run_ars(params)
    weights = trained_policy.weights

