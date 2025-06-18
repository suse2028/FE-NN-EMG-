import numpy as np
import pandas as pd
import torch
import scipy as sp
from scipy import signal
from scipy.fft import fft, fftshift
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import math


##NN-FE:
#Architechture Breakdown": Data streaming pipeline followed by RNN based regression with Physics and Boundary Condition-based Loss

def bc_loss():
    a, b, x, y = #Find way to periodically define them non-explicitly
    rot_bound_min, rot_bound_max = min(data[x][y]), max(data[x][y])
    trans_bound_min, trans_bound_max = min(data[a],[b]), max(data[a][b])

    n = range(1,11)

    period_condition = []
    for i in range(len(data[0])):
        if i == rot_bound_max * any(n)

    #Continue working on integrating rot and trans constraintd


def physics_loss():
    


#RNN Architecture: Convolutional Layer, Pooling Layer, then Activation Layer

class LSTM():




#Maximum Probability Estimation:




#Weight Initialization for Network Call:
tensor = (32, 1)
i_weights = nn.init.xavier_uniform(tensor)
i_biases = Tensor.to(np.zeros(tensor))


