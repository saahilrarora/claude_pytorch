'''
Exercise 2: Reshaping & Indexing

Create a 1D tensor of values 0 through 11 (hint: torch.arange)
Reshape it into a 3x4 tensor
Extract the second row
Extract the element at row 1, column 2
Extract the last column of all rows using slice notation
'''

import torch

# Create a 1D tensor of values 0 through 11 (hint: torch.arange)
tens = torch.arange(12)
# Reshape it into a 3x4 tensor
tens = torch.reshape(tens, (3,4))
# Extract the second row
row_2 = tens[1]
# Extract the element at row 1, column 2
element = tens[0][1]
# Extract the last column of all rows using slice notation
last_col = tens[:,-1]


