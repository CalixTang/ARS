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
    Key distinction is that we create use new dimensions
    """

    def __init__(self, policy_params):
        Policy.__init__(self, policy_params)

        #extra parameters - required for having some sense of lifted dimensions
        self.robot_dim = policy_params['robot_dim']
        self.obj_dim = policy_params['obj_dim']

        self.koopman_obser = DraftedObservable(self.robot_dim, self.obj_dim)
        self.weight_dim = self.koopman_obser.compute_observables_from_self()

        #weights will be the koopman matrix - this can be pretty big
        self.weights = np.zeros((self.weight_dim, self.weight_dim), dtype = np.float64)

    #koopman policies will need to perform updates in some custom way
    def act(self, ob):
        return NotImplementedError
    
    #perform the koopman update z_{t+1} = K @ z_t
    def update_lifted_state(self, z):
        return np.dot(self.weights, z)

    def get_weights_plus_stats(self):
        mu, std = self.observation_filter.get_stats()
        aux = np.asarray([self.weights, mu, std])
        return aux

class RelocatePolicy(KoopmanPolicy):
    """
    Linear policy class that computes action as torque output given from <w, ob>. 
    """

    def __init__(self, policy_params):
        KoopmanPolicy.__init__(self, policy_params)

        #we use a pid controller
        self.pid_controller = policy_params['PID_controller']


    #for our case, observation is a dict of the full env state 
    def act(self, ob):
        #extract relevant state information - [hand pos, target pos - obj pos, obj ori, obj vel]
        x = np.concatenate((ob['qpos'][ : 30], ob['target_pos'] - ob['obj_pos'], ob['qpos'][33:36], ob['qvel'][30:36]))
        
        #normalize if V2 - TODO figure out if we need to normalize, i've turned it off by default for now
        x = self.observation_filter(x, update=self.update_filter)

        #extract lifted state from state
        hand_state, obj_state = x[ : self.robot_dim], x[self.robot_dim : ]
        z = self.koopman_obser.z(hand_state, obj_state)

        #koopman update
        next_z = self.update_lifted_state(z)

        #torque action from lifted state
        action = self.get_act_from_lifted_state(next_z, ob)
        
        return action
    
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
#For now, eigenvalues will get a default value of 1 to make thing easier.
class PartialRelocatePolicy(KoopmanPolicy):
    """
    Policy parameters are a set of dynamic modes (eigenvecs/eigvals of K). 
    """

    def __init__(self, policy_params):
        KoopmanPolicy.__init__(self, policy_params)

        #we use a pid controller
        self.pid_controller = policy_params['PID_controller']

        #additional parameter - the number of modes to use. can be up to the number of lifted states
        self.num_modes = policy_params['num_modes']

        #make sure num modes is valid 
        assert self.num_modes > 0
        assert self.num_modes <= self.weight_dim

        #to allow for storing eigenvalues in the same matrix
        self.weight_dim += 1
        #weights in the shape [weight_dim + 1, num_modes], where the extra 1 comes from storing the eigenvalue associated with the mode
        self.weights = np.zeros((self.weight_dim, self.num_modes), dtype = np.float64)
        #by default, set eigenvalues to 1
        self.weights[-1, :] = 1


    #for our case, observation is a dict of the full env state 
    def act(self, ob):
        #extract relevant state information - [hand pos, target pos - obj pos, obj ori, obj vel]
        x = np.concatenate((ob['qpos'][ : 30], ob['target_pos'] - ob['obj_pos'], ob['qpos'][33:36], ob['qvel'][30:36]))
        
        #normalize if V2 - TODO figure out if we need to normalize, i've turned it off by default for now
        x = self.observation_filter(x, update=self.update_filter)

        #extract lifted state from state
        hand_state, obj_state = x[ : self.robot_dim], x[self.robot_dim : ]
        z = self.koopman_obser.z(hand_state, obj_state)

        #koopman update
        next_z = self.update_lifted_state(z)

        #torque action from lifted state
        action = self.get_act_from_lifted_state(next_z, ob)
        
        return action
    
    #Override the default behavior. Still performs the koopman update z_{t+1} = K @ z_t, but needs to calculate the full K matrix first from the number of modes we have
    def update_lifted_state(self, z):
        #modes = eigvecs
        W = self.weights[:-1, :]
        L = np.diag(self.weights[:, -1])
        #z' = Kz = W L W^{+} z
        return np.dot(np.dot(np.dot(W, L), linalg.pinv(W)), z)
    
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
    