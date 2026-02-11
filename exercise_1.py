'''
Exercise 1: Tensor Basics
Create a Python script that does the following:

Create a 1D tensor from a Python list [1, 2, 3, 4, 5]
Create a 3x3 tensor of all zeros
Create a 3x3 tensor of random values (normal distribution)
Perform element-wise multiplication of the random tensor with itself
Print the shape, dtype, and device of each tensor you created
'''

import torch

example_tensor = torch.tensor([1,2,3,4,5])
three_by_three = torch.zeros(3, 3)
rand_three_by_three = torch.rand(3,3)
rand_three_by_three  = torch.mul(rand_three_by_three, rand_three_by_three)
print(rand_three_by_three)
print(example_tensor.shape, three_by_three.shape, rand_three_by_three.shape)
print(example_tensor.dtype, three_by_three.dtype, rand_three_by_three.dtype)
print(example_tensor.device, three_by_three.device, rand_three_by_three.device)