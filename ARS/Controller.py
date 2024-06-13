"""
A clone of KODex controller.py so we don't have to import mjrl
Define the controllers
"""
import numpy as np

class PD():
    def __init__(self, Kp, Kd, setpoint = None):
        self.Kp, self.Kd = Kp, Kd
        self.setpoint = setpoint

    def __call__(self, input, vel_input):  # input: current hand joints
        '''
        Call the PID controller with *input* and calculate and return a control output
        '''
        error = self.setpoint - input
        d_input = vel_input
        self._proportional = self.Kp * error  # P term
        self._derivative = -self.Kd * d_input #/ dt
        return self._proportional + self._derivative
    
    def set_goal(self, setpoint): # set target joint state
        self.setpoint = setpoint

class PID():
    def __init__(self, Kp, Ki, Kd, setpoint = None):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.setpoint = setpoint
        # self._last_input = None
        # self._last_error = None

    def __call__(self, input, vel_input):  # input: current hand joints
        '''
        Call the PID controller with *input* and calculate and return a control output
        '''
        error = self.setpoint - input
        # d_input = input - (self._last_input if (self._last_input is not None) else input)
        d_input = vel_input
        self._proportional = self.Kp * error  # P term
        # self._integral += self.Ki * error * dt
        # self._integral = _clamp(self._integral, self.output_limits)  # Avoid integral windup
        self._derivative = -self.Kd * d_input #/ dt
        output = self._proportional + self._derivative
        # self._last_output = output
        # self._last_input = input
        # self._last_error = error
        return output
    
    def set_goal(self, setpoint): # set target joint state
        self.setpoint = setpoint
