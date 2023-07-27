import torch

import numpy as np
# x = np.zeros(20)
# print(x.shape)

# x = x[:-1,:]
# print(x)
# print(x.shape)
# x = np.pad(x, [[5 + 1, 0]], 'constant')

a = torch.tensor([[[0 for i in range(5)]]])
a.transpose_(1,2)

for i in range(5):
    a[:,i,:] = i

def f(a):
    j = a.argmax(dim = 1).item()
    a = torch.tensor([[[0 for i in range(a.size(1))]]])
    a.transpose_(1,2)
    a[:,j,:] = 1
    return a

print(f(a))