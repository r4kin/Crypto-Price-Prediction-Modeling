import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np

# Function to identify peaks and troughs
def identify_extrema(data, window_size):
    # Finding peaks
    peaks, _ = find_peaks(data['Predicted'], distance=window_size)
    troughs, _ = find_peaks(-data['Predicted'], distance=window_size)

    # Create columns for peaks and troughs flags
    data['is_peak'] = False
    data['is_trough'] = False

    # Mark the peaks and troughs
    data.loc[peaks, 'is_peak'] = True
    data.loc[troughs, 'is_trough'] = True

    return data

# Read the CSV file
predicted_df = pd.read_csv('predicted_btcusd.csv')

# Identify peaks and troughs in a 10-day window
extrema_df = identify_extrema(predicted_df, 10)

# Calculate the total average and 30-day rolling average
total_avg = predicted_df['Predicted'].mean()
rolling_avg_30 = predicted_df['Predicted'].rolling(window=30).mean()

# Filter dates for plotting (only where extrema are identified)
extrema_dates = predicted_df['date'][extrema_df['is_peak'] | extrema_df['is_trough']]

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(predicted_df['date'], predicted_df['Predicted'], label='Predicted Price', color='blue')
plt.plot(predicted_df['date'], rolling_avg_30, label='30-Day Rolling Average', color='orange', linestyle='dotted')
plt.hlines(total_avg, xmin=predicted_df['date'].iloc[0], xmax=predicted_df['date'].iloc[-1], label='Total Average', colors='purple', linestyles='dotted')
plt.scatter(predicted_df['date'][extrema_df['is_peak']], predicted_df['Predicted'][extrema_df['is_peak']], color='green', marker='^', label='Peaks')
plt.scatter(predicted_df['date'][extrema_df['is_trough']], predicted_df['Predicted'][extrema_df['is_trough']], color='red', marker='v', label='Troughs')
plt.title('BTC Price Prediction with Peaks, Troughs, and Averages')
plt.xlabel('Date')
plt.ylabel('Predicted Price')
plt.xticks(extrema_dates, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
