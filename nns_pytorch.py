from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
import time

#assume the dataset is imported and X and Y are created
#Scale all values. X is the input and Y is the output
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
Y = scaler.fit_transform(Y)

#Split the dataset into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)

#Reshape the data. This is an example with 5 inputs and 1 output
#X_train = X_train.reshape([-1,1,5])
#Y_train = Y_train.reshape([-1,1])
#X_test = X_test.reshape([-1,1,5])
#Y_test = Y_test.reshape([-1,1])

import torch
import torch.nn as nn

#transform train and test sets in tensors
X_train, Y_train, X_test, Y_test = map(torch.tensor, (X_train, Y_train, X_test, Y_test))
n, c = X_train.shape
X_train, X_train.shape, Y_train.min(), Y_train.max()

#Reshape the data. This is an example with 5 inputs and 1 output
#X_train = X_train.reshape([-1,1,5])
#Y_train = Y_train.reshape([-1,1])
#X_test = X_test.reshape([-1,1,5])
#Y_test = Y_test.reshape([-1,1])


#set hyper parameters
input_nodes = 5
num_layers = 1
hidden_nodes = 3
output_nodes = 1
batch_size = 100
#set the number of epoch
epochs = 10

###############################################
#create a feed forward back propagation NN - BPNN with pytorch
start = time.time()
model  = nn.Sequential(nn.Linear(input_nodes, hidden_nodes), nn.Tanh(), 
                        nn.Linear(hidden_nodes, output_nodes), nn.Tanh())

#define losses
mae = torch.nn.L1Loss()
def rmseloss(Y_pred,Y_test):
    return torch.sqrt(torch.mean((Y_pred-Y_test)**2))
rmse = rmseloss

#define optimizers 
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#train the NN
for epoch in range(epochs):

    permutation = torch.randperm(X_train.size()[0])

    for i in range(0,X_train.size()[0], batch_size):
        optimizer.zero_grad()

        batches = permutation[i:i+batch_size]
        batch_x, batch_y = X_train[batches], Y_train[batches]

        Y_train_pred = model(batch_x)
        loss = mae(Y_train_pred, batch_y)
    
        loss.backward()
        optimizer.step()
    print("At epoch: ", epoch, "mae: ", loss.item())
print('compilation duration : ', time.time() - start) 

#test the NN
with torch.no_grad():
    Y_test_pred = model(X_test)
    
    mae_test = mae(Y_test_pred, Y_test)
    rmse_test = rmse(Y_test_pred, Y_test)
    
    print("Test MAE: ", mae_test)
    print("Test RMSE: ", rmse_test)
    
    #invert scaling
    pred_initial = scaler.inverse_transform(Y_test_pred)
    test_initial = scaler.inverse_transform(Y_test)
    
    
    
###############################################
#create a recurrent NN - LSTM with pytorch
import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        hidden = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        cell = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        
        out, _ = self.lstm(x, (hidden, cell))  
        out = self.fc(out[:, -1, :])
        return out

start = time.time()
model = RNN(input_nodes, hidden_nodes, num_layers, output_nodes)

#define losses
mae = torch.nn.L1Loss()
def rmseloss(Y_pred,Y_test):
    return torch.sqrt(torch.mean((Y_pred-Y_test)**2))
rmse = rmseloss

#define optimizers
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#train the RNN
for epoch in range(epochs):

    permutation = torch.randperm(X_train.size()[0])

    for i in range(0,X_train.size()[0], batch_size):
        optimizer.zero_grad()

        batches = permutation[i:i+batch_size]
        batch_x, batch_y = X_train[batches], Y_train[batches]

        Y_train_pred = model(batch_x)
        loss = mae(Y_train_pred, batch_y)
    
        loss.backward()
        optimizer.step()
    print("At epoch: ", epoch, "mae: ", loss.item())

print('compilation duration : ', time.time() - start) 

#test the RNN
with torch.no_grad():
    Y_test_pred = model(X_test)
    loss_test = mae(Y_test_pred, Y_test)
    rmse_test = rmse(Y_test_pred, Y_test)
    
    print("Test MAE: ", loss_test)
    print("Test RMSE: ", rmse_test)

    #invert scaling
    pred_initial = scaler.inverse_transform(Y_test_pred)
    test_initial = scaler.inverse_transform(Y_test)

