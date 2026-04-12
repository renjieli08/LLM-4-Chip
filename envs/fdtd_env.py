"""
custom gym environment developed for invoking FDTD simulations used in RL/BO.
Author: Renjie Li. March 2025 @ NOEL.
"""

from src import FdtdRlNanobeam
import random
import gym
from gym import spaces, logger
#from gym import utils
#from gym.utils import seeding
#from collections import namedtuple, deque
#from itertools import count
#import subprocess, time, signal
import numpy as np
#import sys
#import os

# sys.path.append("D:\\Program Files\\Lumerical\\FDTD\\api\\python\\")  # Default windows lumapi path
# sys.path.append(os.path.dirname(__file__))  # Current directory


class FdtdEnv(gym.Env):
    """
    Makes changes to the physical parameters of PCSEL to optimize optical responses.
    Invokes an FDTD session to take in geometric parameters and compute the resulting optical responses.
    """

        # len = 2000E-9  #width
        # t = 450E-9
        # t_1 = 100E-9
        # t_2 = 0
        # t_3 = 315E-9
        # t_4 = 5000E-9
        # n_1 = 3.2035
        # n_2 = 3.4038 #GaAs (from https://refractiveindex.info)
        # n_3 = 3.415
        # n_4 = 3.2035
        # leng = 0.52  #radius
        # leng2 = 0.406 #for triangular holes
        # a = 400E-9


    metadata = {'render.modes': ['human']}

    def __init__(self):
        # limits for net geometrical changes (states). Less important variables are commented out.
        # self.maxDeltaLen = 1000  # 2000E-9  #width
        # self.maxDeltaT = 100    # 450E-9
        # self.maxDeltaT1 = 100   # 100E-9
        # #self.maxDeltaT2 = 0
        # self.maxDeltaT3 = 100   # 315E-9
        # #self.maxDeltaT4 = 5000E-9   #cladding layer, no need to change
        # self.maxDeltaN1 = 0.15  # 3.2035
        # #self.maxDeltaN2 = 3.4038 #GaAs, don't need to change this item
        # self.maxDeltaN3 = 0.15  # 3.415  active layer
        # #self.maxDeltaN4 = 3.2035  #this is equal to self.maxDeltaN1
        # self.maxDeltaLeng = 0.3    #0.52   #radius
        # #leng2 = 0.406 #for triangular holes
        # self.maxDeltaA = 100   # 400E-9
        
        # best geometrical shift values found so far (taken to be initial values)
        self.len = 0.  # 2000E-9  #width
        self.t = 0.   # 450E-9
        self.t1 = -0.   # 100E-9
        self.t3 = -0.   # 315E-9
        self.n1 = 0.  # 3.2035
        self.n3 = 0.  # 3.415 
        self.leng = -0.   #0.52   #radius
        self.a = 0.   # 400E-9
        
        # optimization goal
        self.Q_goal = 2.0e+6  
        self.area_goal = 3.6e-13  #area >= 3.6e-13 m^2
        self.lam_goal = 1310.0   #or 980 nm
        self.P_goal = 0.3    #output power/injecting power >= 30% 
        self.div_goal = 1.0    # divergence angle <= 1 degree

        self.action_space = spaces.Discrete(16)
        self.observation_space = spaces.Box(-15, 15)
        
        #other setup
        #self.seed()
        self.state = None

    def step(self, X):
        #err_msg = "%r (%s) invalid" % (action, type(action))
        #assert self.action_space.contains(action), err_msg

        self.state = X  #update state
        netDLen, netDT, netDT1, netDT3, netDN1, netDN3, netDLeng, netDA = self.state  #update design parameters

        # perform an action in fdtd and compute Q factor
        FR = FdtdRlNanobeam()
        c = 1e-9  # define conversion from m to nm
        Qf, lam, power, area, div_angle = FR.adjustdesignparams(netDLen*c, netDT*c, netDT1*c, netDT3*c, netDN1, netDN3, netDLeng, netDA*c)


        gamma = 1
        eps = 1 
        beta = 1
        alpha = 1
        eta = 1
        # calculate the score

        r1 = gamma * (1 - (self.Q_goal - Qf) / self.Q_goal)
        r2 = eps * (1 - abs(self.lam_goal - lam) / self.lam_goal)
        r3 = beta * (1 - (self.area_goal - area) / self.area_goal)
        r4 = alpha * (1 - (self.P_goal- power) / self.P_goal)
        r5 = eta * (1 + (self.div_goal - div_angle) / self.div_goal)

        r_total = r1 + r2 + 100*r3 + 100*r4 + 20*r5  #for weighted sum 
        score = np.float64(r_total)

        obj = np.array([r1, r2, r3, r4, r5], dtype=np.float64)   #for multi objective

        print('\nQ factor: {:.3f}, resonance lambda: {:.2f}, area: {:.4e}, power: {:.4f}, divergence: {:.4f}\n'.format(Qf, lam, area, power, div_angle))

        return obj, score

    def reset(self):
        # self.state = np.zeros((4,), dtype=np.float32)
        self.state = (self.len, self.t, self.t1, self.t3, self.n1, self.n3, self.leng, self.a)
        return np.array(self.state, dtype=np.float64)








