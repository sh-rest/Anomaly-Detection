import numpy as np
import matplotlib.pyplot as plt #type: ignore
from collections import deque

def z_score_detection(data_point, history, window_sum, window_sq_sum, z_threshold, window_size = 30):
    """
    Detect short-term anomalies using Z-Score with incremental mean and standard deviation.

    Formula:
    ----
    Z = (X - μ) / σ  
    """
        
    if len(history) < window_size:
        history.append(data_point)
        window_sum += data_point
        window_sq_sum += data_point**2
        return False, window_sum, window_sq_sum  # Not enough data yet

    old_point = history.popleft()  # Remove the oldest data point
    window_sum += data_point - old_point  # Update window sum
    window_sq_sum += data_point**2 - old_point**2  # Update window squared sum
    history.append(data_point)  # Add new point to history

    mean = window_sum / window_size
    variance = (window_sq_sum / window_size) - mean**2
    std_dev = np.sqrt(variance)

    if std_dev == 0:
        return False, window_sum, window_sq_sum  # Avoid division by zero
        
    z_score = (data_point - mean) / std_dev
    return abs(z_score) > z_threshold, window_sum, window_sq_sum

def ema_detection(data_point: float, 
                  ema: float, 
                  ema_sd: float, 
                  alpha: float, 
                  k: float) -> tuple[bool, float, float]: # for long-term anomalies
    '''
    Detects long-term anomalies using the Exponential Moving Average (EMA) method.

    Formula:
    ----
    EMA_t = a * X_t + (1 - a) * EMA_(t-1)
    '''
    
    if ema is None: # initialise EMA to the first data point with zero deviation
        ema = data_point  
        ema_sd = 0 

    ema = alpha * data_point + (1 - alpha) * ema # adjust the ema on entry of the new point
    ema_sd = alpha * abs(data_point - ema) + (1 - alpha) * ema_sd # similar to normal ema, but to calculate the moving deviation
    deviation = abs(data_point - ema) - k * ema_sd # checking if the data point deviates significantly from the EMA
    return deviation > 0, ema, ema_sd

def detect_anomalies(data_stream: np.ndarray, 
                     window_size: int = 30, # size of sliding window for Z-score
                     z_threshold: float = 3,  # threshold above which a data point is considered an anomaly
                     alpha: float = 0.1,  # smoothing factor for EMA
                     k: float = 2, # no of standard deviations from the EMA for detecting long-term anomalies
                     long_term_threshold: int = 3 # min no of consecutive long-term anomalies to define a long-term anomaly region
                     ) -> tuple[list[int], list[list[int]], float]:
    """
    Detect anomalies in a data stream using Z-Score for short-term anomalies 
    and Exponential Moving Average (EMA) for long-term anomalies.

    This function identifies short-term anomalies by calculating the Z-Score of 
    each data point within a sliding window, and it detects long-term anomalies 
    by comparing the data points against an EMA that accounts for recent changes 
    in the data.

    Notes:
    ------
    - Short-term anomalies are identified by calculating the Z-Score of each data point within a sliding window. 
      If the absolute Z-Score exceeds the specified threshold, the data point is flagged as an anomaly.
    
    - Long-term anomalies are identified by continuously updating an EMA and measuring how far the current data point deviates from this EMA.
      If the deviation exceeds a certain threshold (based on the standard deviation of the EMA), it indicates a significant change in the data trend.

    """


    #initialising starting parameters and arrays
    short_term_anomalies = []
    long_term_anomaly_regions = []
    ema = None
    ema_sd = 0
    history = deque(maxlen=window_size)  

    long_term_flag = False
    current_region = []

    window_sum = 0
    window_sq_sum = 0
    means = np.zeros(len(data_stream))
    for i, data_point in enumerate(data_stream):
        
        is_short_term_anomaly, window_sum, window_sq_sum = z_score_detection(data_point, history, window_sum, window_sq_sum, z_threshold, window_size)
        if is_short_term_anomaly:  #adding short term anomalies
            short_term_anomalies.append(i)
            
        means[i] = window_sum/window_size

        
        is_long_term_anomaly, ema, ema_sd = ema_detection(data_point, ema, ema_sd, alpha, k) # EMA with SD for long-term anomalies

        if is_long_term_anomaly: # adding regions which are anomalous according to the EMA with SD which we computed above
            current_region.append(i)  
            long_term_flag = True  
        else:
            if long_term_flag:
                long_term_flag = False
                if len(current_region) >= long_term_threshold: # saving region only if it meets the threshold
                    long_term_anomaly_regions.append(current_region)
                current_region = []  # reset current region

    if long_term_flag and len(current_region) >= long_term_threshold: #if anomaly exists in the end
        long_term_anomaly_regions.append(current_region)
    
    return short_term_anomalies, long_term_anomaly_regions, means

def plot_anomalies(data_stream:np.ndarray, 
                   short_term_anomalies: list, 
                   long_term_anomaly_regions:list,
                   means:np.ndarray,
                   window_size:int  = 30):
    """
    Plots the anomalies with the data stream.
    Gives an all time view of anomalies

    """
    plt.figure(figsize=(12, 6))
    plt.plot(data_stream, label="Data Stream", color="blue", linewidth = 1)
    plt.plot(means, label = f"Moving Mean [{window_size}]", color = "lightgreen", linewidth = 2)
    
    #Highlight short-term anomalies
    for index in short_term_anomalies:
        plt.scatter(index, data_stream[index], color="red", label="Short-term Anomalies (Spikes)" if index == short_term_anomalies[0] else "")

    # Highlight long-term anomaly regions
    for region in long_term_anomaly_regions:
        if isinstance(region, list) and len(region) > 0:
            plt.fill_between(range(region[0], region[-1] + 1), min(data_stream), max(data_stream), 
                             color='orange', alpha=0.3,
                             label='Long-term Anomaly Region' if region[0] == long_term_anomaly_regions[0][0] else "")

    plt.title("Anomaly Detection in Data Stream")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

def real_time_plot(data_stream, short_term_anomalies, long_term_anomaly_regions, means, window = True, window_maxlen = 100):
    """
    Real-time plot that shows the data stream, pre-detected short-term anomalies (Z-Score), long-term anomalies (EMA),
    and the moving mean in real-time.
    """

    #window should be made false if you want to see the whole plot and not just a window of the plot of length = window_maxlen

    stream_length = len(data_stream)
    data_points = deque(maxlen=window_maxlen) if window else [] # Sliding window for real-time display or simple array if 'window' is False

    fig, ax = plt.subplots()
    line, = ax.plot([], [], label="Data Stream", color="blue", linewidth=1)
    mean_line, = ax.plot([], [], label="Mean", color="lightgreen", linewidth=2)

    
    ax.set_ylim(min(data_stream) - 1, max(data_stream) + 1) # setting the limits for the y-axis

    for i in range(stream_length):
        new_x = i
        new_y = data_stream[i]

        
        data_points.append((new_x, new_y)) # adding the new data point to the deque

        
        x_values = [x for x, y in data_points] # Extract x and y values from the deque
        y_values = [y for x, y in data_points]

        # Update the main data stream line
        line.set_data(x_values, y_values)

        if i < len(means):
            mean_line.set_data(range(i + 1), means[:i + 1])

        # Highlight short-term anomalies
        if i in short_term_anomalies:
            ax.scatter(i, data_stream[i], color="red", label="Short-term Anomalies (Spikes)" if i == short_term_anomalies[0] else "")

        # Highlight long-term anomaly regions
        for region in long_term_anomaly_regions:
            plt.fill_between(region, min(data_stream), max(data_stream), color='orange', alpha=0.3,
                label='Long-term Anomaly Region' if 'Long-term Anomaly Region' not in plt.gca().get_legend_handles_labels()[1] else "")
        
        # Dynamically adjust the x-axis to keep the plot centered on recent data points 
        # or to increase length of the x-axis with the new data based on 'window'
        if(window):
            ax.set_xlim(max(0, i - 100), i + 1)
        else:
            ax.set_xlim(0, max(100, i+1))
        # Redraw the plot in real-time
        plt.pause(0.00000001)
        plt.legend()

    # Show the final plot after the loop finishes
    plt.show()


#Example usage
if __name__ == "__main__":
    np.random.seed(42)
    stream_length = 1000
    data_stream = np.random.normal(loc=0.0, scale=1.0, size=stream_length).tolist()
    
    #introducing anomalies
    data_stream[200] = 10 
    data_stream[400:420] = [8] * 20 
    data_stream[600] = -10 
    data_stream[800:820] = [-8] * 20  
    
    # Run anomaly detection
    short_term, long_term, means = detect_anomalies(np.array(data_stream))

    plot_anomalies(data_stream, short_term_anomalies= short_term, long_term_anomaly_regions = long_term, means = means)
    # real_time_plot(data_stream, short_term_anomalies= short_term, long_term_anomaly_regions = long_term, means = means, window = False)
    #  ^ uncomment if real_time_plot needs to be tested