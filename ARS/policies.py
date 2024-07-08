'''
Policy class for computing action from weights and observation vector. 
Horia Mania --- hmania@berkeley.edu
Aurelia Guy
Benjamin Recht 
'''


import numpy as np
from filter import get_filter
from Observables import *
from Controller import *
import scipy.linalg as linalg 

def get_policy(policy_name, policy_params):
    policy_name = policy_name.lower()
    print(policy_name)

    policy = None 
    if policy_name == 'linear':
        policy = LinearPolicy(policy_params)
    elif policy_name == 'koopman':
        policy = KoopmanPolicy(policy_params) 
    elif policy_name == 'truncatedkoopman':
        policy = TruncatedKoopmanPolicy(policy_params)
    elif policy_name == 'minkoopman':
        policy = MinKoopmanPolicy(policy_params)
    elif policy_name == 'relocate':
        policy = RelocatePolicy(policy_params)
    else:
        raise NotImplementedError
    
    if policy_params.get('policy_checkpoint_path', None) is not None:
        print(f"Loading policy weights from {policy_params['policy_checkpoint_path']}")
        policy.update_weights(np.load(policy_params['policy_checkpoint_path'], allow_pickle = True))
    if policy_params.get('filter_checkpoint_path', None) is not None:
        print(f"Loading policy observation filter weights from {policy_params['filter_checkpoint_path']}")
        filter = np.load(policy_params['filter_checkpoint_path'], allow_pickle = True)
        policy.observation_filter = policy.observation_filter.from_dict(filter) 
    
    print(policy.get_weights().shape)
    print(policy_params)
    return policy

class Policy(object):

    def __init__(self, policy_params):

        #modify to allow default values
        self.ob_dim = policy_params.get('ob_dim', -1)
        self.ac_dim = policy_params.get('ac_dim', -1)
        self.weights = np.empty(0)

        # a filter for updating statistics of the observations and normalizing inputs to the policies
        self.observation_filter = get_filter(policy_params['ob_filter'], shape = (self.ob_dim,))
        self.update_filter = True
        
    def update_weights(self, new_weights):
        self.weights[:] = new_weights[:]
        return

    def get_weights(self):
        return self.weights

    def get_observation_filter(self):
        return self.observation_filter

    def act(self, ob):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError

class LinearPolicy(Policy):
    """
    Linear policy class that computes action as <w, ob>. 
    """

    def __init__(self, policy_params):
        Policy.__init__(self, policy_params)
        self.weights = np.zeros((self.ac_dim, self.ob_dim), dtype = np.float64)

    def act(self, ob):
        state = self.extract_state_from_ob(ob)
        state = self.observation_filter(state, update=self.update_filter)
        return np.dot(self.weights, state)
    
    def extract_state_from_ob(self, ob):
        if isinstance(ob, dict):
            # gymnasium robotics envs typically return observations as a dict with robot obs, and then state obs that include object information. we need object info, so we will concatenate everything together to get the actual state
            return np.concatenate([ob['observation'], ob['achieved_goal'], ob['desired_goal']], axis = -1)
        
        return ob

    def get_weights_plus_stats(self):
        
        mu, std = self.observation_filter.get_stats()
        aux = (self.weights, mu, std)
        return aux
  

class KoopmanPolicy(Policy):
    """
    Generic policy for koopman operator-based policies. This version allows for passing specific position and velocity indices to extract from the observation to use in PD control.
    Key distinction between Koopman Policy and Linear Policy:
        Koopman learns the koopman matrix and needs a separate conversion from next lifted state to action
        Linear learns a direct (state -> action) function
    """

    def __init__(self, policy_params):
        super().__init__(policy_params)

        #lifting function + inferred weight dimension
        self.koopman_obser = get_observable(policy_params['lifting_function'])(policy_params['ob_dim'])
        self.weight_dim = self.koopman_obser.compute_observables_from_self()

        #weights will be the koopman matrix - this can be pretty big
        self.weights = np.eye(self.weight_dim, dtype = np.float64)
        self.pid_controller = policy_params['PID_controller']

        self.pos_idx = policy_params.get('obs_pos_idx', np.r_[:])
        self.vel_idx = policy_params.get('obs_vel_idx', np.r_[:])

    #Act is not intended to be overridden - the functions called inside should be overriden
    def act(self, ob):
        #extract relevant state information - usually done when the environment gives us more variables in the observaiton than we really need
        x = self.extract_state_from_ob(ob)
        
        #normalize if running ARS-V2
        x_norm = self.observation_filter(x, update=self.update_filter)

        #extract lifted state from state - z = g(x)
        z = self.extract_lifted_state(x_norm)

        #koopman update - z_{t + 1} = K @ z_t
        next_z = self.update_lifted_state(z)

        #get action from lifted state
        action = self.get_act_from_lifted_state(next_z, x)
        
        return action
    
    def get_act_from_lifted_state(self, next_z, env_state):
        
        #assume that z contains x at its front
        next_x = next_z[:self.ob_dim]

        #get the next position to use as setpoint for controller
        next_pos = next_x[self.pos_idx]
        self.pid_controller.set_goal(next_pos)

        #extract current position and velocity too
        curr_pos, curr_vel = env_state[self.pos_idx], env_state[self.vel_idx]

        #assume that pid will convert angle and angular velocity to torque
        torque_action = self.pid_controller(curr_pos, curr_vel)
        return torque_action
    
    def extract_state_from_ob(self, ob):
        if isinstance(ob, dict):
            # gymnasium robotics envs typically return observations as a dict with robot obs, and then state obs that include object information. we need object info, so we will concatenate everything together to get the actual state
            return np.concatenate([ob['observation'], ob['achieved_goal'], ob['desired_goal']], axis = -1)
        
        return ob
    
    def extract_lifted_state(self, x):
        return self.koopman_obser.z(x)
    
    def update_lifted_state(self, z):
        return self.weights @ z
    
    def get_weights_plus_stats(self):
        mu, std = self.observation_filter.get_stats()
        aux = (self.weights, mu, std)
        return aux

class TruncatedKoopmanPolicy(KoopmanPolicy):
    """
    Implements a truncated representation of the Koopman matrix K. 
    Instead of keeping a dxd matrix for weights, keep a mxd matrix for weights, where m is the state/observation size.
    """
    def __init__(self, policy_params):
        super().__init__(policy_params)

        self.koopman_obser = get_observable(policy_params['lifting_function'])(policy_params['ob_dim'])
        self.weight_dim = self.koopman_obser.compute_observables_from_self()

        self.state_dim = policy_params.get('state_dim', policy_params['ob_dim'])

        #weights will be the truncated koopman matrix
        self.weights = np.eye(self.state_dim, self.weight_dim, dtype = np.float64)
        self.pid_controller = policy_params['PID_controller']

        self.pos_idx = policy_params['obs_pos_idx']
        self.vel_idx = policy_params['obs_vel_idx']

    def get_act_from_lifted_state(self, next_z, env_state):
        #for this class, next_z is the next x

        #get the next position to use as setpoint for controller
        next_pos = next_z[self.pos_idx]
        self.pid_controller.set_goal(next_pos)

        #extract current position and velocity too
        curr_pos, curr_vel = env_state[self.pos_idx], env_state[self.vel_idx]

        #assume that pid will convert angle and angular velocity to torque
        torque_action = self.pid_controller(curr_pos, curr_vel)
        # print(torque_action)
        return torque_action

class MinKoopmanPolicy(KoopmanPolicy):
    """
        Implements a minimal representation of the Koopman matrix K. 
        Instead of keeping a dxd matrix for weights, keep an axd matrix for weights, where a is the action size (but the output is the relevant next states (positions)). 
        This is similar to a Linear Policy which maps state to action, except we are going from lifted state to action size to (through PID) action.
    """
    def __init__(self, policy_params):
        super().__init__(policy_params)

        self.koopman_obser = get_observable(policy_params['lifting_function'])(policy_params['ob_dim'])
        self.weight_dim = self.koopman_obser.compute_observables_from_self()

        self.state_dim = policy_params.get('state_dim', policy_params['ob_dim'])

        #weights will be the minimized koopman matrix - we have to initialize these using state pos idx
        self.weights = np.zeros((self.ac_dim, self.weight_dim), dtype = np.float64)
        for i in range(self.pos_idx.shape[0]):
            self.weights[i, self.pos_idx[i]] = 1

        self.pid_controller = policy_params['PID_controller']

        self.pos_idx = policy_params['obs_pos_idx']
        self.vel_idx = policy_params['obs_vel_idx']

    def get_act_from_lifted_state(self, next_z, env_state):
        #for this class, next_z is the next x

        #next_z is in the size of the action dimension. we can use it directly for minkoopman
        self.pid_controller.set_goal(next_z)

        #extract current position and velocity too
        curr_pos, curr_vel = env_state[self.pos_idx], env_state[self.vel_idx]

        #assume that pid will convert angle and angular velocity to torque
        torque_action = self.pid_controller(curr_pos, curr_vel)
        # print(torque_action)
        return torque_action

class EigenKoopmanPolicy(KoopmanPolicy):
    """
    Implements a compressed representation of the Koopman matrix K = W L W^+ (eigendecomposition representation)
    Weights contains the eigenvectors and eigenvalues. An update to weights will update a hidden internal koopman matrix.
    """

    def __init__(self, policy_params):
        super().__init__(policy_params)

        #the number of modes (num of eigenvectors/values) can be up to the number of lifted states.
        self.num_modes = policy_params['num_modes']
        self.clip_eigvals = policy_params.get('clip_eigvals', False)

        #make sure num modes is valid 
        assert self.num_modes > 0

        # can't use this because usually actual weight dim isnt set up yet
        # assert self.num_modes <= self.weight_dim

        #to allow for storing eigenvalues in the same matrix
        self.weight_dim += 1
        #weights in the shape [weight_dim + 1, num_modes], where the extra 1 comes from storing the eigenvalue associated with the mode
        self.weights = np.eye(self.weight_dim, self.num_modes, dtype = np.float64)
        
        #by default, set eigenvalues to 1
        self.weights[-1, :] = 1
        self.koopman_mat = self.koopman_mat_from_weights()
    
    def update_weights(self, new_weights):

        #if this is a koopman matrix, we update differently
        if new_weights.shape[1] == self.weight_dim - 1:
            L, W = linalg.eig(new_weights)
            L = L[ : self.num_modes].real
            W = W[:, : self.num_modes].real

            self.weights[: -1, :] = W
            self.weights[-1, :] = L
        else:
            self.weights[:] = new_weights[:]
        
        if self.clip_eigvals:
            self.weights[-1, :] = np.clip(self.weights[-1, :], 0, 1) #clip eigenvalues to be reasonable values
        
        #update internal koopman mat from weights to avoid duplicate matmuls during rollouts
        self.koopman_mat = self.koopman_mat_from_weights()
    
    def koopman_mat_from_weights(self):
        #modes = eigvecs
        W = self.weights[:-1, :]
        L = np.diag(self.weights[-1, :])
        #K = W L W^{+}
        return W @ L @ linalg.pinv(W)
    
    #Override the default behavior. Still performs the koopman update z_{t+1} = K @ z_t
    def update_lifted_state(self, z):
        #modes = eigvecs
        #z' = Kz = W L W^{+} z
        return self.koopman_mat @ z


class RelocatePolicy(KoopmanPolicy):
    """
    Koopman Policy class for the DAPG relocate task.
    """

    def __init__(self, policy_params):
        super().__init__(policy_params)

        #extra parameters - required for having some sense of lifted dimensions
        self.robot_dim = policy_params['robot_dim']
        self.obj_dim = policy_params['obj_dim']

        self.koopman_obser = ManipulationObservable(self.robot_dim, self.obj_dim)
        self.weight_dim = self.koopman_obser.compute_observables_from_self()

        self.weights = np.eye(self.weight_dim, self.weight_dim, dtype = np.float64)

    def extract_state_from_ob(self, ob):
        return np.concatenate((ob['qpos'][ : 30], ob['obj_pos'] - ob['target_pos'], ob['qpos'][33:36], ob['qvel'][30:36]))
        # ob['qpos'][30:33]
    def extract_lifted_state(self, x):
        hand_state, obj_state = x[ : self.robot_dim], x[self.robot_dim : ]
        return self.koopman_obser.z(hand_state, obj_state)
    
    def get_act_from_lifted_state(self, next_z, env_state):
        #gym_env.py from KODex/CIMER mujoco
        next_hand_state, next_obj_state = next_z[:self.robot_dim], next_z[2 * self.robot_dim: 2 * self.robot_dim + self.obj_dim]  # retrieved robot & object states
        self.pid_controller.set_goal(next_hand_state)

        #TODO: figure out if pid control is the right way to go about this
        torque_action = self.pid_controller(env_state['qpos'][ : self.robot_dim], env_state['qvel'][ : self.robot_dim])
        return torque_action
        


#TODO: figure out potential design problem: we might expect good learned eigenvalues to take on a different range than the values of eigenvectors, but ARS will treat them the same in exploration.
#TODO: we are using A = W L W^+ as a reconstruction tactic, but I haven't proved that this is mathematically sound when W isn't nxn 
#This is also a very messy way of "eigendecomposition" - we could try to force eigvecs to be orthonormal but that sounds like a lot of work
#For now, eigenvalues will get a default value of 1 to make thing easier.
# class EigenRelocatePolicy(EigenKoopmanPolicy):
#     """
#     Policy parameters are a set of dynamic modes (eigenvecs/eigvals of K). 
#     """

#     def __init__(self, policy_params):
#         super().__init__(policy_params)

#         #extra parameters - required for having some sense of lifted dimensions
#         self.robot_dim = policy_params['robot_dim']
#         self.obj_dim = policy_params['obj_dim']

#         self.koopman_obser = ManipulationObservable(self.robot_dim, self.obj_dim)
#         self.weight_dim = self.koopman_obser.compute_observables_from_self() + 1

#         self.weights = np.eye(self.weight_dim, self.num_modes, dtype = np.float64)
#         self.weights[-1, :] = 1
#         self.koopman_mat = self.koopman_mat_from_weights()

#     def extract_state_from_ob(self, ob):
#         return np.concatenate((ob['qpos'][ : 30], ob['obj_pos'] - ob['target_pos'], ob['qpos'][33:36], ob['qvel'][30:36]))
        
#     def extract_lifted_state(self, x):
#         hand_state, obj_state = x[ : self.robot_dim], x[self.robot_dim : ]
#         return self.koopman_obser.z(hand_state, obj_state)
    
#     def get_act_from_lifted_state(self, next_z, env_state):
#         #gym_env.py from KODex/CIMER mujoco
#         next_hand_state, next_obj_state = next_z[:self.robot_dim], next_z[2 * self.robot_dim: 2 * self.robot_dim + self.obj_dim]  # retrieved robot & object states
#         self.pid_controller.set_goal(next_hand_state)

#         #TODO: figure out if pid control is the right way to go about this
#         torque_action = self.pid_controller(env_state['qpos'][ : self.robot_dim], env_state['qvel'][ : self.robot_dim])
#         return torque_action



# TODO: think about designing ARS-K policies in a way where we only randomize rows that matter for PD control. This may be unfair though.