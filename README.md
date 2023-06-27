# Crypto-Price Prediction-Modeling


Introduction -------------

This is an exploratory research project to compare the performance of statistical modeling techniques with machine learning models for price prediction on multivariate cryptocurrencies. The Jupyter notebook, "project.ipynb", outlines the Python code used to implement the ARIMA statistical model, and the deep-learning LSTM and GRU models using PyTorch. It was found comparing the RMSE values of the models, that the GRU model outperformed the ARIMA and LSTM models. 

Note: The cryptocurrency data used is multivariate and highly volatile which will highly impact the conclusions drawn from this analysis


Method -------------

- Historical market data for BTC, ETH, and XRP coins were collected from online repositories. This data was then cleaned and analyzed to identify open-to-close returns and close price moving averages
- Data was tested for non-stationarity using the Augmented Dicky Fuller (ADF) and KPSS test, seeking p-value
- Data was found to be non-stationary, and thus first order differencing and log transforming were applied
- Further analysis was done by plotting the Autocorrelation and Partial Autocorrelation Functions (ACF/PACF), seeking d and q-values
- Fit our ARIMA model based on determined p,d, and q values and evaluated the RMSE value for the prediction model     
- Implemented the testing and training of LSTM and GRU models using PyTorch by converting data into tensors
- Compared RMSE values of the three models to determine their price prediction performance 
 




