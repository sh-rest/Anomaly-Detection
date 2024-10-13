import numpy as np
import streamSim as ss
import anomalyAlgo as aa


np.random.seed(42)
stream_length = 1000  
data_stream = ss.simDataStream(stream_length)

short_term, long_term, means = aa.detect_anomalies(data_stream, window_size=30, z_threshold=3, alpha=0.1, k=2, long_term_threshold=3)

# aa.plot_anomalies(data_stream, short_term, long_term, means)
aa.real_time_plot(data_stream,short_term, long_term, means, window = False)
