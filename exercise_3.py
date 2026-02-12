'''
Exercise 3: Autograd Basics
This is where PyTorch starts to feel different from NumPy. Autograd is the engine behind all neural network training.

Create a tensor x = torch.tensor([2.0, 3.0], requires_grad=True)
Compute y = x ** 2 + 3 * x + 1
Compute z = y.sum()
Call z.backward()
Print x.grad and verify (by hand or in a comment) that the gradient is correct
'''
import torch

x = torch.tensor([2.0, 3.0], requires_grad=True) 
y = (x ** 2) + (3 * x) + 1
z = y.sum()
z.backward()
print(x.grad)