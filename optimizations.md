# Optimizations in Anomaly Detection Algorithms

## 1. Z-Score

A Z-Score is a statistical measure that indicates how many standard deviations a data point is from the mean of the data set. It’s commonly used to detect anomalies or outliers in a dataset.

- **Definition**: A Z-Score quantifies how far a particular data point is from the average in terms of standard deviations.
- **Threshold**: Typically, values beyond a certain threshold (e.g., a Z-Score greater than 3 or less than -3) are considered anomalous, as they fall far outside the typical range of values.

## 2. Exponential Moving Average (EMA) with Standard Deviation

An Exponential Moving Average (EMA) is a type of moving average that gives more weight to recent data points, making it more responsive to changes in the data compared to a simple moving average (SMA).

- **Functionality**: By calculating the EMA of a data stream, we can smooth out short-term fluctuations and detect gradual changes or trends over time.
- **Incorporation of SD**: Adding the standard deviation (SD) as a monitoring factor helps identify when the variation around the EMA is unusually high or low, indicating potential anomalies.

## 3. Why Choose This Combination of Algorithms?

- **Z-Score for Spikes**: This algorithm detects large, immediate deviations (spikes) from the current pattern. If the data suddenly jumps to a value far outside the normal range, the Z-Score will flag it as an anomaly.
- **EMA with SD for Gradual Drift**: The EMA adapts over time, allowing it to detect gradual trends or drifts in the data. As the EMA tracks the average and the SD tracks variability, it can identify when data starts to deviate slowly from the norm over a longer period, which might go unnoticed by Z-Score alone.

## 4. Other Optimizations

### Use of Deques for Sliding Window

Implementing `deque` from the `collections` module for the sliding window in the `z_score_detection` function allows for efficient O(1) operations for appending and popping elements. This improves performance over using a standard list.

### Incremental Updates

The `z_score_detection` function updates the window sum and window squared sum incrementally instead of recalculating them from scratch for each new data point. This change reduces computational overhead, bringing the complexity of finding the mean and standard deviation from O(N) to O(1).

### Efficient Long-term Anomaly Detection

The `ema_detection` function initializes the Exponential Moving Average (EMA) to the first data point and uses a recursive update method for the EMA and its standard deviation. This approach allows for continuous adaptation to changing data trends and reduces computational overload since only a single operation needs to take place using the previous value of the EMA.
