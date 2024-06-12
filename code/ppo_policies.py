"""
Contains policies used by PPO. We define a Koopman-based policy suitable for use as an actor network and a normal NN policy suitable for a critic network.
"""
import torch
import torch.nn.functional as F
import numpy as np

import Observables
# from Controller import PID
	
class KoopmanNetworkPolicy(torch.nn.Module):
	def __init__(self, obs_dim, act_dim, observable_class, state_pos_idx, state_vel_idx, controller_P, controller_D):
		"""
			Initialize policy and set up params  
			Parameters
				obs_dim - input dimensions as int (M)
				act_dim - output dimensions as int (A)
				observable_class - the observable class object containing the lifting function 
				controller_P - a PD controller's P gain
				controller_D - a PD controller's D gain
		"""
		super(KoopmanNetworkPolicy, self).__init__()

		self.obs_dim, self.act_dim = obs_dim, act_dim

		#instantiate observable (has lifting function)
		self.observable = observable_class(obs_dim)
		
		#size of the lifted state (D)
		self.lifted_dim = self.observable.compute_observables_from_self()
		
		#(truncated) koopman matrix. of size [A, D]. We do not want a bias term.
		self.koopman_layer = torch.nn.Linear(self.lifted_dim, self.obs_dim, bias = False)

		#TODO - figure out if this initialization is necessary or useful
		#initialize koopman layer's weights to identity instead of the default Linear random init b/c of koopman update
		with torch.no_grad():
			self.koopman_layer.weight.copy_(torch.nn.Parameter(torch.eye(self.obs_dim, self.lifted_dim)))
		
		#PD control layer
		self.PD_layer = PDControlLayer(self.obs_dim, self.act_dim, state_pos_idx, state_vel_idx, controller_P, controller_D)
		#Important - we do not want to update the weights in the PD layer
		for param in self.PD_layer.parameters():
			param.requires_grad = False

	def forward(self, obs):
		"""
			Implements a forward pass of the policy. For the Koopman policy, turns an env observation into an action
		"""

		#1) lift observable to lifted state (R^M -> R^D)
		z = self.observable.z(obs)

		#2) Use Koopman matrix (aka linear layer) to propagate lifted state to next state
		x_prime = self.koopman_layer(z)

		#3) Use PD control module to generate action - pass in curr obs and x_prime as setpoint
		return self.PD_layer(obs, x_prime)
	

"""
An implementation of a PD controller using nn.Module. This is so that we can perform gradient ascent with gradient flow through the PD controller. 
"""
class PDControlLayer(torch.nn.Module):
	def __init__(self, obs_dim, act_dim, state_pos_idx, state_vel_idx, P, D):
		'''
			Initialize PD Control Layer values. 
			Implementation assumes that all current position and velocity values are within the observation.

			Parameters:
				obs_dim - the observation dimension (M)
				act_dim - the action dimension (A)
				state_pos_idx - an list/ndarray/tensor containing, in order, the indices of position variables within the observation. Of shape (A, )
				state_vel_idx - an list/ndarray/tensor containing, in order, the indices of velocity variables within the observation. Of shape (A, )
				P - the proportional gain constant of the PD controller
				D - the derivative gain constant of the PD controller
		'''
		super(PDControlLayer, self).__init__()

		assert act_dim == state_pos_idx.shape[0]
		assert act_dim == state_vel_idx.shape[0]

		self.obs_dim = obs_dim
		self.state_pos_idx, self.state_vel_idx = state_pos_idx, state_vel_idx
		self.P, self.D = P, D

		self.pos_extract_layer = torch.nn.Linear(obs_dim, act_dim, bias = False)
		self.vel_extract_layer = torch.nn.Linear(obs_dim, act_dim, bias = False)

		self.setup_extract_layers()


	def setup_extract_layers(self):
		'''
			Helper function for __init__. 
			Sets up the position extraction and velocity extraction layers based on their supplied indices in the observation space.
			Note: this formulation assumes that the pos and vel indices are the same in both observation space AND lifted space (this is always true if the lifted space just has the observation space in the front)
		'''

		with torch.no_grad():
			self.pos_extract_layer.weight.copy_(torch.nn.Parameter(torch.zeros_like(self.pos_extract_layer.weight)))
			self.vel_extract_layer.weight.copy_(torch.nn.Parameter(torch.zeros_like(self.vel_extract_layer.weight)))

			#TODO: find a better way to do this - I couldn't figure out if it's possible to set a bunch of elements inline if they're not a contiguous slice
			for i in range(self.state_pos_idx.shape[0]):
				self.pos_extract_layer.weight[i, self.state_pos_idx[i]] = 1
				self.vel_extract_layer.weight[i, self.state_vel_idx[i]] = 1

	def forward(self, obs, next_state):
		'''
			The forward function of a PD controller. Requires current observation (including pos and velocity information) and next state. I use full next state for ease of implementation

			Parameters
				obs - The full current observation. Of shape (obs_dim, )
				next_state - The full next state. Of shape (obs_dim, )
		'''
		#extract curr pos, vel, and setpoint
		pos = self.pos_extract_layer(obs)
		vel = self.vel_extract_layer(obs)
		setpoint = self.pos_extract_layer(next_state)
		
		#proportional gain = kP * (setpoint - curr pos)
		p_gain = self.P * (setpoint - pos)

		#derivative gain = -kD * curr vel
		d_gain = -self.D * vel

		#final res = P gain + (no I gain) + D gain
		return p_gain + d_gain


"""
This is a direct copy of the FeedForwardNN from ppo_for_beginners. 

Original code: https://github.com/ericyangyu/PPO-for-Beginners/blob/master/part4/ppo_for_beginners/network.py
"""
class NNPolicy(torch.nn.Module):
	"""
		A standard in_dim-64-64-out_dim Feed Forward Neural Network.
	"""
	def __init__(self, in_dim, out_dim):
		"""
			Initialize the network and set up the layers.

			Parameters:
				in_dim - input dimensions as an int
				out_dim - output dimensions as an int

			Return:
				None
		"""
		super(NNPolicy, self).__init__()

		self.layer1 = torch.nn.Linear(in_dim, 64)
		self.layer2 = torch.nn.Linear(64, 64)
		self.layer3 = torch.nn.Linear(64, out_dim)

	def forward(self, obs):
		"""
			Runs a forward pass on the neural network.

			Parameters:
				obs - observation to pass as input

			Return:
				output - the output of our forward pass
		"""
		# Convert observation to tensor if it's a numpy array
		if isinstance(obs, np.ndarray):
			obs = torch.tensor(obs, dtype=torch.float)

		activation1 = F.relu(self.layer1(obs))
		activation2 = F.relu(self.layer2(activation1))
		output = self.layer3(activation2)

		return output
	
#For Unit testing - remove when done
# if __name__ == '__main__':
# 	# x = torch.ones((10))
# 	batch_x = torch.cat((torch.ones((1, 10)), 2 * torch.ones(1, 10)), axis = 0)


# 	model = KoopmanNetworkPolicy(10, 5, Observables.LocomotionObservableTorch, torch.tensor([4, 3, 2, 1, 0]), torch.tensor([9, 8, 7, 6, 5]), 1, 0.1)
# 	criterion = torch.nn.MSELoss(reduction='sum')
# 	optimizer = torch.optim.SGD(model.parameters(), lr=1) #toy example
	
# 	model.train()
# 	y = model(batch_x)
# 	print(y, flush = True) # [-0.1 x 5]

# 	y_pred = 10 * torch.ones((2, 5)) #just so that we'll have noticeable steps
# 	loss = criterion(y_pred, y)
# 	print(loss, flush = True)

# 	optimizer.zero_grad()
# 	loss.backward()
# 	optimizer.step()

# 	print([param for param in model.parameters()])



