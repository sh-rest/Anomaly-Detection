# Anomaly Detection in Data Streams

## Overview

This project implements an anomaly detection system for data streams using two statistical methods: Z-Score for short-term anomalies and Exponential Moving Average (EMA) for long-term anomalies. The system identifies anomalies in real-time, allowing for immediate insights into data behavior.

## Features

- **Short-term Anomaly Detection**: Uses Z-Score to detect spikes and anomalies in recent data points within a sliding window.
- **Long-term Anomaly Detection**: Utilizes Exponential Moving Average (EMA) to track trends and identify significant deviations over time.
- **Real-time Plotting**: Visualizes data streams along with detected anomalies and moving averages in real-time.
- **Data Visualization**: Generates plots for a comprehensive view of the data stream and detected anomalies.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/sh-rest/anomaly-detection.git
   cd anomaly-detection
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Anomaly Detection

You can run the anomaly detection script using Python.

```bash
python main.py
```

### Input Data

The project generates a synthetic data stream with added anomalies for demonstration. You can modify the `data_stream` parameters in the `streamSim.py` file to test different scenarios.
There are several components to the dataStream being generated

- base sine wave
- seasonal wave
- noise
- line with a positive/negative slope with a cutoff function

### Anomaly Detection Functions

- `detect_anomalies(data_stream, window_size, z_threshold, alpha, k, long_term_threshold)`: Detects short-term and long-term anomalies in the provided data stream.

- `plot_anomalies(data_stream, short_term_anomalies, long_term_anomaly_regions, means, window_size)`: Plots the data stream along with identified short-term and long-term anomalies.

- `real_time_plot(data_stream, short_term_anomalies, long_term_anomaly_regions, means, window, window_maxlen)`: Displays the data stream in real-time with detected anomalies and moving averages.

## Example Output

After running the script, you will see a plot displaying:

- The original data stream (in blue).
- The moving mean (in light green).
- Short-term anomalies (marked in red).
- Long-term anomaly regions (highlighted in orange).

## Customization

- You can adjust parameters such as `window_size`, `z_threshold`, `alpha`, `k`, and `long_term_threshold` in the `detect_anomalies` function call to tailor the detection sensitivity to your needs.
- You can also adjust parameters such as `window` and `window_maxlen` in the `real_time_plot` function call to tailor the view in which you receive the final plot

[Go to documentation](optimizations.md)
