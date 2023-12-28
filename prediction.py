import pandas as pd
import numpy as np
import yfinance as yf
import os
import matplotlib.pyplot as plt
import time
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

btc_data = pd.read_csv(os.getcwd() + '/Data/BTCUSD.csv')

btc_data.rename(columns={'Volume BTC': 'Volume'}, inplace=True)

btc_data = btc_data[btc_data['date'] >= '2017-12-07'].iloc[::-1].reset_index(drop=True)

for df in [btc_data]:
    df.drop('unix', axis = 1, inplace = True)
    df['close_MA21'] = df['close'].rolling(window=21).mean().fillna(method='backfill')
    df['close_MA14'] = df['close'].rolling(window=14).mean().fillna(method='backfill')
    df['close_MA10'] = df['close'].rolling(window=21).mean().fillna(method='backfill')
    df['gk_vol'] = np.sqrt(252*((np.log(df['high']/df['low']))**2) - (2*np.log(2) - 1)*((np.log(df['close']/df['open']))**2))
    df['o2c_ret'] = np.log(df['close']/df['open'])

scaler = MinMaxScaler(feature_range=(-1,1)) # scale time-series between -1 and 1
btc_data['norm_close'] = scaler.fit_transform(btc_data['close'].values.reshape(-1,1))

# Load data for each crypto
def load_data(crypto, look_back):
    data_raw = crypto[['norm_close']].to_numpy()
    data = []

    # create all possible sequences of length look_back
    for index in range(len(data_raw) - look_back): 
        data.append(data_raw[index: index + look_back])
        
    data = np.array(data) # makes it 2d

    # Split into test-train sets
    test_size = int(np.round(0.2*data.shape[0]))
    train_size = data.shape[0] - test_size
    

    x_train = data[:train_size,:-1,:] # splitting is done here
    y_train = data[:train_size,-1,:]

    x_test = data[train_size:,:-1]
    y_test = data[train_size:,-1,:]

    return [x_train, y_train, x_test, y_test]

# GRU Implementation
class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

lookback = 60
x_train, y_train, x_test, y_test = load_data(btc_data, lookback)

# Convert data into tensors for Pytorch
x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train = torch.from_numpy(y_train).type(torch.Tensor)
y_test = torch.from_numpy(y_test).type(torch.Tensor)


# Parameters, we can tune this later
input_dim = 1
output_dim = 1
hidden_dim = 32
num_layers = 3
learning_rate = 0.01

# Initialize model, loss function and optimizer
model = GRU(input_dim=input_dim, hidden_dim=hidden_dim,  num_layers=num_layers, output_dim=output_dim)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# Train Model
num_epochs = 50
hist = np.zeros(num_epochs)
start_time = time.time()

for t in range(num_epochs):
    
    # Forward pass
    y_train_pred = model(x_train)
    loss = loss_fn(y_train_pred, y_train)
    
    if t % 10 == 0 and t !=0:
        print("Epoch", t, "MSE: ", np.around(loss.item(), 5))

    hist[t] = loss.item() # Store results

    # Optimize, and zero out gradient
    optimizer.zero_grad()
    # Backward pass
    loss.backward()
    # Update parameters
    optimizer.step()

training_time = time.time()-start_time
print("Training time: {} seconds".format(np.around(training_time,2)))

# Make Predictions
y_test_pred = model(x_test)

# invert predictions
y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
y_train = scaler.inverse_transform(y_train.detach().numpy())
y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
y_test = scaler.inverse_transform(y_test.detach().numpy())

# calculate root mean squared error
trainScore = np.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = np.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


test_data = btc_data.iloc[-len(y_test_pred):]
predicted_df = test_data[['date', 'symbol']].copy()
predicted_df['Predicted'] = y_test_pred
predicted_df.to_csv('predicted_btcusd.csv', index=False)
