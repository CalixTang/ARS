'''
Policy class for computing action from weights and observation vector. 
Horia Mania --- hmania@berkeley.edu
Aurelia Guy
Benjamin Recht 
'''


import numpy as np
from filter import get_filter
from mjrl.KODex_utils.Observables import *
from mjrl.KODex_utils.Controller import *
import scipy.linalg as linalg 

def get_policy(policy_name, policy_params):
    if policy_name == 'linear':
        return LinearPolicy(policy_params)
    elif policy_name == 'relocate':
        return RelocatePolicy(policy_params)
    elif policy_name == 'koopman':
        return KoopmanPolicy(policy_params)
    elif policy_name == 'cheetah':
        return MinCheetahPolicy(policy_params)
    elif policy_name == 'swimmer':
        return MinSwimmerPolicy(policy_params)
    elif policy_name == 'eigenrelocate':
        return EigenRelocatePolicy(policy_params)
    else:
        raise NotImplementedError

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
        ob = self.observation_filter(ob, update=self.update_filter)
        return np.dot(self.weights, ob)

    def get_weights_plus_stats(self):
        
        mu, std = self.observation_filter.get_stats()
        aux = np.asarray([self.weights, mu, std])
        return aux
        
class KoopmanPolicy(Policy):
    """
    General policy for koopman operator-based policies.
    Key distinction between Koopman Policy and Linear Policy:
        Koopman learns the koopman matrix and needs a separate conversion from next lifted state to action
        Linear learns a direct (state -> action) function
    """

    def __init__(self, policy_params):
        super().__init__(policy_params)

        #lifting function + inferred weight dimension
        self.koopman_obser = IdentityObservable(policy_params['ob_dim'])
        self.weight_dim = self.koopman_obser.compute_observables_from_self()

        #weights will be the koopman matrix - this can be pretty big
        self.weights = np.zeros((self.weight_dim, self.weight_dim), dtype = np.float64)
        self.pid_controller = policy_params['PID_controller']

    #Act is not intended to be overridden - the functions called inside should be overriden
    def act(self, ob):
        #extract relevant state information - usually done when the environment gives us more variables in the observaiton than we really need
        x = self.extract_state_from_ob(ob)
        
        #normalize if running ARS-V2
        x = self.observation_filter(x, update=self.update_filter)

        #extract lifted state from state - z = g(x)
        z = self.extract_lifted_state(x)

        #koopman update - z_{t + 1} = K @ z_t
        next_z = self.update_lifted_state(z)

        #get action from lifted state
        action = self.get_act_from_lifted_state(next_z, ob)
        
        return action

    
    def extract_state_from_ob(self, ob):
        return ob
    
    def extract_lifted_state(self, x):
        return self.koopman_obser.z(x)
    
    #perform the koopman update z_{t+1} = K @ z_t
    def update_lifted_state(self, z):
        return self.weights @ z
    
    def get_act_from_lifted_state(self, next_z, env_state):
        #assume that z contains x at its front
        next_x = next_z[:self.ob_dim]
        #assume that env_state is x
        self.pid_controller.set_goal(next_x)
        return self.pid_controller(env_state)



    def get_weights_plus_stats(self):
        mu, std = self.observation_filter.get_stats()
        aux = np.asarray([self.weights, mu, std])
        return aux
    
class EigenKoopmanPolicy(KoopmanPolicy):
    """
    General policy for koopman operator-based policies.
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
        assert self.num_modes <= self.weight_dim

        #to allow for storing eigenvalues in the same matrix
        self.weight_dim += 1
        #weights in the shape [weight_dim + 1, num_modes], where the extra 1 comes from storing the eigenvalue associated with the mode
        self.weights = np.zeros((self.weight_dim, self.num_modes), dtype = np.float64)
        
        #by default, set eigenvalues to 1
        self.weights[-1, :] = 1
        self.koopman_mat = self.koopman_mat_from_weights()
    
    def update_weights(self, new_weights):
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

    def get_weights_plus_stats(self):
        mu, std = self.observation_filter.get_stats()
        aux = np.asarray([self.weights, mu, std])
        return aux
    
class SwimmerPolicy(KoopmanPolicy):
    """
    Swimmer v2 policy
    """

    def __init__(self, policy_params):
        super().__init__(policy_params)

        self.koopman_obser = LocomotionObservable(policy_params['ob_dim'])
        self.weight_dim = self.koopman_obser.compute_observables_from_self()
        
        #testing out diff instantiation
        self.weights = np.zeros((self.weight_dim, self.weight_dim), dtype = np.float64)
    
    #Should be overridden
    def get_act_from_lifted_state(self, next_z, env_state):
        #follow Half Cheetah v2 specs

        #assume that z contains x at its front
        next_x = next_z[:self.ob_dim]
        #joint angles (next states)
        next_pos = next_x[1 : 3]

        #assume that env_state is x
        self.pid_controller.set_goal(next_pos)

        curr_pos, curr_vel = env_state[1 : 3], env_state[-2 :]

        #assume that pid will convert angle and angular velocity to torque
        torque_action = self.pid_controller(curr_pos, curr_vel)
        # print(torque_action)
        return torque_action

    def get_weights_plus_stats(self):
        mu, std = self.observation_filter.get_stats()
        aux = np.asarray([self.weights, mu, std])
        return aux
    
class MinSwimmerPolicy(KoopmanPolicy):
    """
    Another Swimmer v2 policy
    """

    def __init__(self, policy_params):
        super().__init__(policy_params)

        self.koopman_obser = LocomotionObservable(2)
        self.ob_dim = 2
        self.weight_dim = self.koopman_obser.compute_observables_from_self()
        
        #testing out diff instantiation
        self.weights = np.zeros((self.weight_dim, self.weight_dim), dtype = np.float64)
    
    #get the relevant joint angles only - don't even look at angular velocities
    def extract_state_from_ob(self, ob):
        return ob[1 : 3]

    def get_act_from_lifted_state(self, next_z, env_state):
        #follow Half Cheetah v2 specs

        #assume that z contains x at its front
        next_pos = next_z[:self.ob_dim]

        #assume that env_state is x
        self.pid_controller.set_goal(next_pos)

        curr_pos, curr_vel = env_state[1 : 3], env_state[-2 :]

        #assume that pid will convert angle and angular velocity to torque
        torque_action = self.pid_controller(curr_pos, curr_vel)
        return torque_action

    def get_weights_plus_stats(self):
        mu, std = self.observation_filter.get_stats()
        aux = np.asarray([self.weights, mu, std])
        return aux
    


class CheetahPolicy(KoopmanPolicy):
    """
    Cheetah v2 policy
    """

    def __init__(self, policy_params):
        super().__init__(policy_params)

        self.koopman_obser = LocomotionObservable(policy_params['ob_dim'])
        self.weight_dim = self.koopman_obser.compute_observables_from_self()
        
        #testing out diff instantiation
        self.weights = np.zeros((self.weight_dim, self.weight_dim), dtype = np.float64)
    
    #Should be overridden
    def get_act_from_lifted_state(self, next_z, env_state):
        #follow Half Cheetah v2 specs

        #assume that z contains x at its front
        next_x = next_z[:self.ob_dim]
        #joint angles (next states)
        next_pos = next_x[2:8]

        #assume that env_state is x
        self.pid_controller.set_goal(next_pos)

        curr_pos, curr_vel = env_state[2:8], env_state[11:17]

        #assume that pid will convert angle and angular velocity to torque
        torque_action = self.pid_controller(curr_pos, curr_vel)
        # print(torque_action)
        return torque_action

    def get_weights_plus_stats(self):
        mu, std = self.observation_filter.get_stats()
        aux = np.asarray([self.weights, mu, std])
        return aux
    
class MinCheetahPolicy(KoopmanPolicy):
    """
    Another Cheetah v2 policy
    """

    def __init__(self, policy_params):
        super().__init__(policy_params)

        self.koopman_obser = LocomotionObservable(6)
        self.ob_dim = 6
        self.weight_dim = self.koopman_obser.compute_observables_from_self()
        
        #testing out diff instantiation
        self.weights = np.zeros((self.weight_dim, self.weight_dim), dtype = np.float64)
    
    #get the relevant joint angles only - don't even look at angular velocities
    def extract_state_from_ob(self, ob):
        return ob[2:8]

    def get_act_from_lifted_state(self, next_z, env_state):
        #follow Half Cheetah v2 specs

        #assume that z contains x at its front
        next_pos = next_z[:self.ob_dim]

        #assume that env_state is x
        self.pid_controller.set_goal(next_pos)

        curr_pos, curr_vel = env_state[2:8], env_state[11:17]

        #assume that pid will convert angle and angular velocity to torque
        torque_action = self.pid_controller(curr_pos, curr_vel)
        # print(torque_action)
        return torque_action

    def get_weights_plus_stats(self):
        mu, std = self.observation_filter.get_stats()
        aux = np.asarray([self.weights, mu, std])
        return aux
    

class RelocatePolicy(KoopmanPolicy):
    """
    Linear policy class that computes action as torque output given from <w, ob>. 
    """

    def __init__(self, policy_params):
        super().__init__(policy_params)

        #extra parameters - required for having some sense of lifted dimensions
        self.robot_dim = policy_params['robot_dim']
        self.obj_dim = policy_params['obj_dim']

        self.koopman_obser = ManipulationObservable(self.robot_dim, self.obj_dim)
        self.weight_dim = self.koopman_obser.compute_observables_from_self()

        self.weights = np.zeros((self.weight_dim, self.weight_dim), dtype = np.float64)

    def extract_state_from_ob(self, ob):
        return np.concatenate((ob['qpos'][ : 30], ob['target_pos'] - ob['obj_pos'], ob['qpos'][33:36], ob['qvel'][30:36]))
    
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
        
    def get_weights_plus_stats(self):
        
        mu, std = self.observation_filter.get_stats()
        aux = np.asarray([self.weights, mu, std])
        return aux
    

#TODO: figure out potential design problem: we might expect good learned eigenvalues to take on a different range than the values of eigenvectors, but ARS will treat them the same in exploration.
#TODO: we are using A = W L W^+ as a reconstruction tactic, but I haven't proved that this is mathematically sound when W isn't nxn 
#This is also a very messy way of "eigendecomposition" - we could try to force eigvecs to be orthonormal but that sounds like a lot of work
#For now, eigenvalues will get a default value of 1 to make thing easier.
class EigenRelocatePolicy(EigenKoopmanPolicy):
    """
    Policy parameters are a set of dynamic modes (eigenvecs/eigvals of K). 
    """

    def __init__(self, policy_params):
        super().__init__(policy_params)

        #extra parameters - required for having some sense of lifted dimensions
        self.robot_dim = policy_params['robot_dim']
        self.obj_dim = policy_params['obj_dim']

        self.koopman_obser = ManipulationObservable(self.robot_dim, self.obj_dim)
        self.weight_dim = self.koopman_obser.compute_observables_from_self() + 1

        self.weights = np.zeros((self.weight_dim, self.num_modes), dtype = np.float64)
        self.weights[-1, :] = 1
        self.koopman_mat = self.koopman_mat_from_weights()

    def extract_state_from_ob(self, ob):
        return np.concatenate((ob['qpos'][ : 30], ob['target_pos'] - ob['obj_pos'], ob['qpos'][33:36], ob['qvel'][30:36]))
        
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

        
    def get_weights_plus_stats(self):
        
        mu, std = self.observation_filter.get_stats()
        aux = np.asarray([self.weights, mu, std])
        return aux


# TODO: brainstorm what it would mean to implement a SVD reconstruction-based policy