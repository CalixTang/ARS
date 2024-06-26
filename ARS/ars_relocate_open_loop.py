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
from mjrl.utils.gym_env import GymEnv
from mjrl.KODex_utils.Controller import *
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
import graph_results
import gym


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
        # initialize policy 
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

        _ = self.env.reset()
        robot_dim, obj_dim = self.policy_params['robot_dim'], self.policy_params['obj_dim']
        
        ob = self.env.get_env_state()
        x = np.concatenate((ob['qpos'][ : 30], ob['obj_pos'] - ob['target_pos'], ob['qpos'][33:36], ob['qvel'][30:36]))
        
        hand_state, obj_state = x[ : robot_dim], x[robot_dim : ]
        z = self.policy.koopman_obser.z(hand_state, obj_state)

        success_ctr = 0
        success_thresh = 10

        for i in range(rollout_length):
            next_z = self.policy.update_lifted_state(z)

            next_hand_state, next_obj_state = next_z[:robot_dim], next_z[2 * robot_dim: 2 * robot_dim + obj_dim]  # retrieved robot & object states
            self.policy.pid_controller.set_goal(next_hand_state) 

            reward = 0
            done = False
            goal_achieved = False

            for _ in range(5):
                sub_ob = self.env.get_env_state()
                # for relocation task, it we set a higher control frequency, we can expect a much better PD performance
                torque_action = self.policy.pid_controller(sub_ob['qpos'][:30], sub_ob['qvel'][:30])
                torque_action[1] -= 0.95  # hand_Txyz[1] -> hand_T_y
                next_o, substep_reward, done, goal_achieved = self.env.step(torque_action) 

                reward += substep_reward
            
            if goal_achieved['goal_achieved']:
                success_ctr += 1


            """
            #generate torque action
            action = self.policy.act(ob)
            reward = 0

            #TODO: check if this is valid - we are doing 1 koopman update - 1 step here, and in CIMER it's handled as 1 koopman - 5 steps
            next_o, reward, done, goal_achieved = self.env.step(action)  
            """

            z = next_z
            
            # ob, reward, done, _ = self.env.step(action)
            steps += 1
            total_reward += (reward - shift)
            if done:
                break
            
        return total_reward, steps, success_ctr > success_thresh

    def do_rollouts(self, w_policy, num_rollouts = 1, shift = 1, evaluate = False):
        """ 
        Generate multiple rollouts with a policy parametrized by w_policy.
        """

        rollout_rewards, deltas_idx, successes = [], [], []
        steps = 0

        for i in range(num_rollouts):

            if evaluate:
                self.policy.update_weights(w_policy)
                deltas_idx.append(-1)
                
                # set to false so that evaluation rollouts are not used for updating state statistics
                self.policy.update_filter = False

                # for evaluation we do not shift the rewards (shift = 0) and we use the
                # default rollout length (1000 for the MuJoCo locomotion tasks)
                reward, r_steps, success = self.rollout(shift = 0., rollout_length = self.rollout_length)
                rollout_rewards.append(reward)
                successes.append(success)
                
            else:
                idx, delta = self.deltas.get_delta(w_policy.size)
             
                delta = (self.delta_std * delta).reshape(w_policy.shape)
                deltas_idx.append(idx)

                # set to true so that state statistics are updated 
                self.policy.update_filter = True

                # compute reward and number of timesteps used for positive perturbation rollout
                self.policy.update_weights(w_policy + delta)
                pos_reward, pos_steps, pos_success = self.rollout(shift = shift)

                # compute reward and number of timesteps used for negative pertubation rollout
                self.policy.update_weights(w_policy - delta)
                neg_reward, neg_steps, neg_success = self.rollout(shift = shift) 
                steps += pos_steps + neg_steps

                rollout_rewards.append([pos_reward, neg_reward])
                successes.append([pos_success, neg_success])
                            
        return {'deltas_idx': deltas_idx, 'rollout_rewards': rollout_rewards, "steps" : steps, "successes": successes}
    
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
        self.filter_type = policy_params['ob_filter']

        
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

        rollout_rewards, deltas_idx, successes = [], [], [] 

        for result in results_one:
            if not evaluate:
                self.timesteps += result["steps"]
            deltas_idx += result['deltas_idx']
            rollout_rewards += result['rollout_rewards']
            successes += result['successes']

        for result in results_two:
            if not evaluate:
                self.timesteps += result["steps"]
            deltas_idx += result['deltas_idx']
            rollout_rewards += result['rollout_rewards']
            successes += result['successes']

        deltas_idx = np.array(deltas_idx)
        rollout_rewards = np.array(rollout_rewards, dtype = np.float64)
        successes = np.array(successes).flatten()
        t2 = time.time()
        print('Mean reward of collected rollouts:', rollout_rewards.mean())
        print('Maximum reward of collected rollouts:', rollout_rewards.max())
        print('Success rate of collected rollouts: ', successes.mean())
        

        print('Time to generate rollouts:', t2 - t1)
        raw_rollout_rewards = rollout_rewards[:]

        if evaluate:
            return rollout_rewards, successes

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
                rewards, successes = self.aggregate_rollouts(num_rollouts = num_eval_rollouts, evaluate = True)
                w = ray.get(self.workers[0].get_weights_plus_stats.remote())
                # np.save(self.logdir + "/koopman_policy.npy", w)

                eval_rewards[i // 10, :] = rewards.flatten()
                if rewards.mean() > best_eval_policy_reward:
                    best_eval_policy_reward = rewards.mean()
                    best_eval_policy_weights = w[0]
                    if self.filter_type == 'MeanStdFilter':
                        best_filter_mean, best_filter_std = w[1], w[2]
                
                #eval logging
                # print(sorted(self.params.items()))
                logz.log_tabular("Time", time.time() - start)
                logz.log_tabular("Iteration", i + 1)
                logz.log_tabular("AverageReward", np.mean(rewards))
                logz.log_tabular("StdRewards", np.std(rewards))
                logz.log_tabular("MaxRewardRollout", np.max(rewards))
                logz.log_tabular("MinRewardRollout", np.min(rewards))
                logz.log_tabular("Task Success Rate", np.mean(successes))
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

        #save best weights
        print(f'Best eval policy mean reward: {best_eval_policy_reward}')
        np.save(os.path.join(self.logdir, 'best_koopman_policy_weights.npy'), best_eval_policy_weights)
        np.save(os.path.join(self.logdir, 'latest_koopman_policy_weights.npy'), self.w_policy)        
 
        np.save(os.path.join(self.logdir, 'training_rewards.npy'), training_rewards)
        np.save(os.path.join(self.logdir, 'eval_rewards.npy'), eval_rewards)


        if self.filter_type == 'MeanStdFilter':
            ob_filter_obj = self.policy.get_observation_filter().as_dict()
            np.save(os.path.join(self.logdir, 'obs_filter.npy'), ob_filter_obj)

            best_ob_filter_obj = self.policy.get_observation_filter.copy()
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

    #i think we don't care about these for our case?
    # ob_dim = env.observation_space.shape[0]
    # ac_dim = env.action_space.shape[0]
    # print(ob_dim, ac_dim)
    ob_dim, ac_dim = params['robot_dim'] + params['obj_dim'], 0

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
                   'num_modes': params['num_modes'], # only for EigenRelocate policy
                   'PID_controller': Simple_PID,
                   'policy_checkpoint_path': params.get('policy_checkpoint_path', ''),
                   'filter_checkpoint_path': params.get('filter_checkpoint_path', '')
                   }
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
    parser.add_argument('--task_id', type=str, default='relocate')
    parser.add_argument('--n_iter', '-n', type=int, default=2000) #training steps
    parser.add_argument('--n_directions', '-nd', type=int, default=320) #directions explored - results in 2*d actual policies
    parser.add_argument('--deltas_used', '-du', type=int, default=80) #directions kept for gradient update
    parser.add_argument('--step_size', '-s', type=float, default=0.02)#0.02, alpha in the paper
    parser.add_argument('--delta_std', '-std', type=float, default=0.004)# 0.03, v in the paper
    parser.add_argument('--n_workers', '-e', type=int, default = 8)
    parser.add_argument('--rollout_length', '-r', type=int, default=100) #100 timesteps * 5 b/c of the PID subsampling
    # for Swimmer-v1 and HalfCheetah-v1 use shift = 0
    # for Hopper-v1, Walker2d-v1, and Ant-v1 use shift = 1
    # for Humanoid-v1 used shift = 5
    parser.add_argument('--shift', type=float, default=0) #TODO: tweak as necessary
    parser.add_argument('--seed', type=int, default=237)
    parser.add_argument('--policy_type', type=str, default='eigenrelocate')
    parser.add_argument('--dir_path', type=str, default='data')
    # for ARS V1 use filter = 'NoFilter', V2 = 'MeanStdFilter'
    parser.add_argument('--filter', type=str, default='NoFilter') 

    #relocate-specific arguments 
    parser.add_argument('--object', type=str, default = 'ball')
    parser.add_argument('--robot_dim', type=int, default = 30)
    parser.add_argument('--obj_dim', type=int, default = 12)
    parser.add_argument('--num_modes', type=int, default = 300) #EigenRelocate only, for relocate task in [1, 759]
    
    #utility arguments
    parser.add_argument('--params_path', type = str)
    parser.add_argument('--policy_checkpoint_path', type = str)
    parser.add_argument('--filter_checkpoint_path', type = str)
    parser.add_argument('--run_name', type = str)
    
    
    
    args = parser.parse_args()
    params = vars(args)

    ray.init(num_cpus=args.n_workers)

    if args.params_path is not None:
        import json
        params = json.load(open(args.params_path, 'r'))
        # print(params)
    
    run_ars(params)
