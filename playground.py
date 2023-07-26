import torch

import numpy as np
x = np.zeros(20)
print(x.shape)

x = x[:-1,:]
print(x)
print(x.shape)
x = np.pad(x, [[5 + 1, 0]], 'constant')