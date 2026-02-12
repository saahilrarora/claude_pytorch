'''
Deep Learning with Pytorch 2 for pytorch fundamentals"
https://www.youtube.com/watch?v=2yBEZzQu8dA&list=PLCC34OHNcOtpcgR9LEYSdi9r7XIbpkpK1&index=2
'''

import torch
import numpy as np

# Regular list in python
my_list = [1,2,3,4,5]
print(my_list)

# numpy array
np1 = np.random.rand(3)
print(np1)

# tensor array
tensor_2d = torch.randn(3,4)
print(tensor_2d)

# create tensor out of numpy array
my_tensor = torch.tensor(np1)
print(my_tensor)