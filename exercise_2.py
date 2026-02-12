'''
Deep Learning with Pytorch 2 for pytorch fundamentals -- tensor operations
https://www.youtube.com/watch?v=Q5gK3qDA5Tc&list=PLCC34OHNcOtpcgR9LEYSdi9r7XIbpkpK1&index=3
'''

import torch

my_torch = torch.arange(10)
print(my_torch)

# reshape and view
my_torch = my_torch.reshape(2,5)
print(my_torch)

my_torch2 = torch.arange(10)
print(my_torch2)

# reshape if we dont know the number of items using -1 (2 is the number of rows we want, must be possible with our tensor size)
my_torch2 = my_torch2.reshape(2,-1)
print(my_torch2)

my_torch3 = torch.arange(10)

# view() only works on contiguous tensors and always returns a view (sharing the original data), while reshape() is more flexible: 
# it returns a view if possible, but creates a copy if the data is not contiguous in memory. 
my_torch4 = my_torch3.view(2,5)
print(my_torch4)

# Slices
my_torch7 = torch.arange(10)
# grabs the tensor item
print(my_torch7[7])

my_torch8 = my_torch7.reshape(5,2)
print(my_torch8)
print(my_torch8[:,0])

# return column
print(my_torch8[:,1:])