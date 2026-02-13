'''
Tensor math operations
https://www.youtube.com/watch?v=Ta3z9vZaoMc&list=PLCC34OHNcOtpcgR9LEYSdi9r7XIbpkpK1&index=4
'''
import torch
import numpy as np

tensor_a = torch.tensor([1,2,3,4])
tensor_b = torch.tensor([5,6,7,8])

# element wise addition
print(tensor_a + tensor_b)

# addition using torch
print(torch.add(tensor_a, tensor_b))

#subtraction
print(tensor_b - tensor_a)

# subtraction using torch
print(torch.sub(tensor_b, tensor_a))

# Multiplcation
print(tensor_a*tensor_b)

# mult using torch
print(torch.mul(tensor_a, tensor_b))

# Division
print(tensor_b / tensor_a)

# division using torch
print(torch.div(tensor_b, tensor_a))

# modulo
print(tensor_b%tensor_a)

# modulo using torch
print(torch.remainder(tensor_b, tensor_a))

# Exponents / power
print(tensor_a ** tensor_b)

# exponents using torch
print(torch.pow(tensor_a, tensor_b))

# reassignment
# tensor_a = tensor_a+tensor_b
tensor_a.add_(tensor_b)
print(tensor_a)