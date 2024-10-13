import numpy as np
import matplotlib.pyplot as plt #type: ignore

def simDataStream(stream_length: int)-> np.ndarray:
    """
    Simulates a time series data stream with anomalies.

    Args:
        stream_length (int): The length of the data stream.

    Returns:
        tuple[np.ndarray, np.ndarray]: 
            - time (np.ndarray): Array of time points.
            - data_stream (np.ndarray): Array of simulated data values.
    """
    np.random.seed(42)  # For reproducibility
    
    base_frequency = 0.1  # Frequency of the base sine wave
    seasonal_frequency = 0.01  # Frequency of the seasonal wave
    noise_amplitude = 0.4  # Amplitude of random noise
    anomaly_frequency = 0.05  # Probability of anomaly at each time step
    anomaly_magnitude = 10  # Magnitude of the anomalies
    slope = 0.02 # Slope of ascending trend
    con = 5 # Constant added to line

    time=np.arange(stream_length) # Generate time points
    
    zer = np.zeros(stream_length) # Generate an array of zeroes
    line = slope * time - con # Base pattern: line 
    ascend = np.maximum(zer, line)

    base_pattern=np.sin(2 * np.pi * base_frequency * time) # Base pattern: sine wave
    base_pattern[500:] += 10

    seasonal_component = 0.5 * np.sin(2 * np.pi * seasonal_frequency * time) # Seasonal component: lower frequency sine wave

    random_noise = noise_amplitude * np.random.randn(stream_length) # Random noise

    data_stream = base_pattern + seasonal_component + random_noise + ascend # Combine base pattern, seasonal component, noise and increase if required 


    for i in range(stream_length):        # Inject anomalies
        if np.random.rand() < anomaly_frequency:
            data_stream[i] += anomaly_magnitude * (2 * np.random.rand() - 1)  # Add a large spike or dip

    return data_stream

def plotDataStream(data_stream: np.ndarray) -> None: 
    '''
    Plots the data stream alone
    '''
    plt.figure(figsize=(12, 6))
    plt.plot(data_stream, label="Data Stream", color="blue")
    plt.title("Simulated Data Stream with Anomalies")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    data_stream = simDataStream(1000)
    plotDataStream(data_stream)