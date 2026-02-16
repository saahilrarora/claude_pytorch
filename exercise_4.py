'''
Create a basic neural network
https://www.youtube.com/watch?v=JHWqWIoac2I&list=PLCC34OHNcOtpcgR9LEYSdi9r7XIbpkpK1&index=5

Load Data and Train Neural Network Model
https://www.youtube.com/watch?v=Xp0LtPBcos0&list=PLCC34OHNcOtpcgR9LEYSdi9r7XIbpkpK1&index=6

Evaluate Test Data Set On Network
https://www.youtube.com/watch?v=rgBu8CbH9XY&list=PLCC34OHNcOtpcgR9LEYSdi9r7XIbpkpK1&index=7
'''
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

# Create a model class that inherits the nn.Module
class Model(nn.Module):
    # Input layer (4 features of the flower) --> Hidden layer1 (number of neurons) --> H2 (n) --> output (3 classes of iris flowers)

    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        super().__init__()
        self.fc1 = nn.Linear(in_features,h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Pick a manual seed for randomization
torch.manual_seed(41)

# Create instance of model
model = Model().to(device)

url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
my_df = pd.read_csv(url)

# Change last column from strings to numbers
my_df['species'] = my_df['species'].replace('setosa', 0.0)
my_df['species'] = my_df['species'].replace('versicolor', 1.0)
my_df['species'] = my_df['species'].replace('virginica', 2.0)
#print(my_df)

# Train Test Split! Set X, y
X = my_df.drop('species', axis=1)
y = my_df['species']

# Convert these to numpy arrays
X = X.values
y = y.values.astype(float)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

# Convert X features to float tensors and move to device
X_train = torch.FloatTensor(X_train).to(device)
X_test = torch.FloatTensor(X_test).to(device)

# convert y labels to tensors long and move to device
y_train = torch.LongTensor(y_train).to(device)
y_test = torch.LongTensor(y_test).to(device)

# Set the criterion of model to measure the error, how far off the predictions are from the data
criterion = nn.CrossEntropyLoss()
# Choose Adam Optimizer, lr = learning rate (if error doesn't go down after a bunch of epochs, lower learning rate)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train our model!
# Epochs? (one run through all the training data in our network)
epochs = 100
losses = []
for i in range(epochs):
    # go forward and get a prediction
    y_pred = model.forward(X_train) # get predicted results

    # Measure the loss/error
    loss = criterion(y_pred, y_train) # predicted values vs y_train

    # Keep track of losses
    losses.append(loss.detach().cpu().numpy())
    
    # print every 10 epochs
    if i % 10 == 0:
        print(f'Epoch: {i} and loss: {loss}')
    
    # Do some back propagation: take the error rate of forward propagation and feed it back
    # through the network to fine tune the weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(range(epochs), losses)
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()

# Evaluate model on test data set (validate model on test set)
with torch.no_grad(): # turn off back propogation
    y_eval = model.forward(X_test) # X_test are features from our test set, y_eval will be our predictions
    loss = criterion(y_eval, y_test) # Find the loss or error

print(loss)