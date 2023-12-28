import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
import numpy as np

# Function to identify local maximums and minimums
def identify_extrema(data, window_size):
    # Finding local maxima and minima using argrelextrema
    max_indices = argrelextrema(data['Predicted'].values, comparator=np.greater, order=window_size)[0]
    min_indices = argrelextrema(data['Predicted'].values, comparator=np.less, order=window_size)[0]

    # Create columns for maxima and minima flags
    data['is_max'] = False
    data['is_min'] = False

    # Mark the maxima and minima
    data.loc[max_indices, 'is_max'] = True
    data.loc[min_indices, 'is_min'] = True

    return data

# Read the CSV file
predicted_df = pd.read_csv('predicted_btcusd.csv')

# Identify local maximums and minimums in a 20-day window
extrema_df = identify_extrema(predicted_df, 10)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(predicted_df['date'], predicted_df['Predicted'], label='Predicted Price', color='blue')
plt.scatter(predicted_df['date'][extrema_df['is_max']], predicted_df['Predicted'][extrema_df['is_max']], color='green', marker='^', label='Local Maxima')
plt.scatter(predicted_df['date'][extrema_df['is_min']], predicted_df['Predicted'][extrema_df['is_min']], color='red', marker='v', label='Local Minima')
plt.title('BTC Price Prediction with Local Maxima and Minima')
plt.xlabel('Date')
plt.ylabel('Predicted Price')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
