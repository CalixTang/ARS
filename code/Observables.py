"Clone of KODex Observables.py (relevant portions) to avoid needing to import mjrl and mj_envs"

"""
Base class for an observable. Should lift a state vector to a lifted state vector.
"""
import numpy as np

class Observable(object):
    def __init__(self, num_envStates):
        self.num_states = num_envStates

    """
    Implementation of lifting function
    """
    def z(self, envState):
        raise NotImplementedError
    
    """
    Compute the size of lifted state given size of state 
    """
    def compute_observable(num_states):
        raise NotImplementedError
    
    """
    Compute the size of lifted state given size of state for this instance.
    """
    def compute_observables_from_self(self):
        raise NotImplementedError

class IdentityObservable(Observable):
    def __init__(self, num_envStates):
        super().__init__(num_envStates)

    """
    Implementation of lifting function
    """
    def z(self, envState):
        return envState
    
    """
    Compute the size of lifted state given size of state 
    """
    def compute_observable(num_states):
        return num_states
    
    """
    Compute the size of lifted state given size of state for this instance.
    """
    def compute_observables_from_self(self):
        return self.num_states

class LocomotionObservable(Observable):
    def __init__(self, num_envStates):
        super().__init__(num_envStates)
    
    def z(self, envState):  
        """
        Lifts the environment state from state space to full "observable space' (Koopman). g(x) = z.
        Inputs: environment states
        Outputs: state in lifted space
        """

        #initialize lifted state arr
        obs = np.zeros(self.compute_observables_from_self())
        index = 0

        #for consistency, I keep the same order of functions as DraftedObservable as used in KODex/CIMER

        #x[i]
        obs[index : index + self.num_states] = envState[:]
        index += self.num_states

        #x[i]^2
        obs[index : index + self.num_states] = envState ** 2
        index += self.num_states  

        obs[index : index + self.num_states] = np.sin(envState)
        index += self.num_states

        obs[index : index + self.num_states] = np.cos(envState)
        index += self.num_states

        #x[i] x[j] w/o repetition
        # for i in range(self.num_states):
        #     for j in range(i + 1, self.num_states):
        #         obs[index] = envState[i] * envState[j]
        #         index += 1

        # x[i]^2 x[j] w/ repetition - I removed the x[i]^3 here b/c this block includes x[i]^3
        # for i in range(self.num_states):
        #     for j in range(self.num_states):
        #         obs[index] = (envState[i] ** 2) * envState[j]
        #         index += 1
        return obs
    
    def compute_observable(num_states):
        """
        Returns the number of observables for a given state space.
        """

        #x[i], x[i]^2, x[i] x[j] w/o rep, x[i]^2 x[j]

        return 4 * num_states
        # return 2 * num_states + (num_states * (num_states - 1) // 2) + num_states ** 2
        
    
    def compute_observables_from_self(self):
        """
        Observation functions: original states, original states^2, cross product of hand states
        """
        return LocomotionObservable.compute_observable(self.num_states)
        

"""
A rewrite of DraftedObservable that is faster
"""
class ManipulationObservable(Observable):
    def __init__(self, num_hand_states, num_obj_states):
        super().__init__(num_hand_states + num_obj_states)
        self.num_hand_states = num_hand_states
        self.num_obj_states = num_obj_states

    def z(self, hand_state, obj_state):  
        """
        Inputs: hand states(pos, vel) & object states(pos, vel)
        Outputs: state in lifted space
        Note: velocity is optional
        """
        # hand_state.shape[0] == self.num_hand_states necessary
        # obj_state.shape[0] == self.num_obj_states necessary

        obs = np.zeros(self.compute_observables_from_self())
        index = 0

        obs[index : index + self.num_hand_states] = hand_state[:]
        index += self.num_hand_states

        obs[index : index + self.num_hand_states] = hand_state[:] ** 2
        index += self.num_hand_states

        obs[index : index + self.num_obj_states] = obj_state[:]
        index += self.num_obj_states

        obs[index : index + self.num_obj_states] = obj_state[:] ** 2
        index += self.num_obj_states

        obj = obj_state[:, np.newaxis]
        obs[index : index + (self.num_obj_states * (self.num_obj_states - 1) // 2)] = (obj @ obj.T)[np.triu_indices(self.num_obj_states, k = 1)]
        index += (self.num_obj_states * (self.num_obj_states - 1) // 2)

        hand = hand_state[:, np.newaxis]
        obs[index : index + (self.num_hand_states * (self.num_hand_states - 1) //2)] = (hand @ hand.T)[np.triu_indices(self.num_hand_states, k = 1)]
        index += (self.num_hand_states * (self.num_hand_states - 1) // 2)

        obs[index : index + self.num_hand_states] = hand_state[:] ** 3
        index += self.num_hand_states

        obs[index : index + self.num_obj_states ** 2] = np.flatten((obj**2) @ (obj.T))
        index += self.num_obj_states ** 2
        return obs
    
    def compute_observable(self, num_hand, num_obj):
        """
        Observation functions: original states, original states^2, cross product of hand states
        """
        return int(2 * (num_hand + num_obj) + (num_obj - 1) * num_obj / 2 + (num_hand - 1) * num_hand / 2 + num_hand + num_obj ** 2)  
    
    def compute_observables_from_self(self):
        """
        Observation functions: original states, original states^2, cross product of hand states
        """
        return ManipulationObservable.compute_observable(self.num_hand_states, self.num_obj_states)
