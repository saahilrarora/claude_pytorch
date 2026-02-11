'''
Exercise 2: Reshaping & Indexing

Create a 1D tensor of values 0 through 11 (hint: torch.arange)
Reshape it into a 3x4 tensor
Extract the second row
Extract the element at row 1, column 2
Extract the last column of all rows using slice notation
'''

import torch

tens = torch.arange(12)
tens = torch.reshape(tens, (3,4))
row_2 = tens[1]
element = tens[0][1]
last_col = tens[:,-1]


