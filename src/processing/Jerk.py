import numpy as np
from scipy import signal
from sklearn.cluster import KMeans

class Jerk:

    def __init__(self, x, x2, fs=16, fc=2.64, timeDays=0):
        self.x = np.array(x) if not isinstance(x, np.ndarray) else x
        self.x2 = np.array(x2) if not isinstance(x2, np.ndarray) else x2
        self.fs = fs
        self.fc = fc
        self.timeDays = timeDays
        self.timeRef = np.copy(self.x[:, 0])
        self.filtered_x = None
        self.kmeans_model = None

    def preprocess_time(self):
        """Preprocess time data and segment it if required."""
        if self.timeDays != 0:
            timeSinceStart = (self.timeRef - self.timeRef[0]).astype(float) / (24 * 60 * 60)
            refMax = np.floor(np.max(timeSinceStart) / self.timeDays)
            refBreak = np.arange(0, refMax + 1) * self.timeDays
            refBreak[-1] += self.timeDays

            timeBin = np.digitize(timeSinceStart, refBreak, right=True)
            return timeBin
        return None

    def filter_data(self):
        """Apply a high-pass Butterworth filter to the data."""
        bf_pca = signal.butter(3, self.fc / (0.5 * self.fs), btype='high', output='sos')
        self.filtered_x = signal.sosfiltfilt(bf_pca, self.x[:, 1:4], axis=0)

        # Handle NaN values
        nas = np.isnan(self.filtered_x)
        nas_vector = np.logical_or.reduce(nas, axis=1)
        if np.any(nas_vector):
            print("NAs found and replaced by 0. NA proportion:", np.mean(nas_vector))
            self.filtered_x[nas_vector, :] = 0
        
        # Smoothing using a moving average
        self.filtered_x = np.concatenate(
            (np.zeros((12, 3)), signal.sosfiltfilt(np.ones(25) / 25, self.filtered_x, axis=0), np.zeros((12, 3))),
            axis=0
        )

    def apply_kmeans(self):
        """Apply KMeans clustering to identify high states."""
        self.kmeans_model = KMeans(n_clusters=2).fit(self.filtered_x)
        high_state = np.argmax(self.kmeans_model.cluster_centers_, axis=0)
        logicalHigh = self.kmeans_model.labels_ == high_state
        self.filtered_x = logicalHigh.astype(float)

    def combine_data(self):
        """Combine the processed data."""
        x_combined = np.prod(self.filtered_x, axis=1)
        temp_x = np.column_stack((x_combined, self.timeRef))
        temp_x = np.max(temp_x, axis=0)
        return temp_x[:, np.newaxis]

    def process(self):
        """Run the full process and return the final result."""
        self.preprocess_time()
        self.filter_data()
        self.apply_kmeans()
        return self.combine_data()




import numpy as np
import pandas as pd
from scipy import signal
from sklearn.cluster import KMeans

class PreyCatchAttemptBehaviours:
    def __init__(self, x, x2, fs=16, fc=2.64, timeDays=0):
        """
        Initialize the object with necessary parameters and data.
        
        Args:
        - x: Pandas DataFrame containing columns 'time', 'ax', 'ay', 'az', 'segmentID'.
        - x2: segmentID identifying continuous time segments.
        - fs: Sampling frequency of signal (default is 16Hz).
        - fc: Frequency above which signals across x, y, and z axes are retained.
        - timeDays: Time period over which KMeans clustering should be performed.
                    Leave as zero to have no time grouping in KMeans generation.
        """
        if not isinstance(x, pd.DataFrame):
            raise ValueError("Input x must be a Pandas DataFrame")
        self.x = x
        self.x2 = x2
        self.fs = fs
        self.fc = fc
        self.timeDays = timeDays
        self.timeRef = self.x['time'].copy()

    def preprocess_time(self):
        """Preprocess the time data and create time bins if required."""
        if self.timeDays != 0:
            timeSinceStart = (self.timeRef - self.timeRef.iloc[0]).dt.total_seconds() / (24 * 60 * 60)
            refMax = np.floor(timeSinceStart.max() / self.timeDays)
            refBreak = np.arange(0, refMax + 1) * self.timeDays
            refBreak = np.append(refBreak, refBreak[-1] + self.timeDays)

            self.x['timeBin'] = np.digitize(timeSinceStart, refBreak, right=False)
            del refBreak, timeSinceStart

    def apply_high_pass_filter(self):
        """Apply a high-pass Butterworth filter to the data."""
        sos = signal.butter(3, self.fc / (0.5 * self.fs), btype='high', output='sos')
        
        def filter_func(col):
            return signal.sosfiltfilt(sos, col)
        
        self.x[['ax', 'ay', 'az']] = self.x.groupby(self.x2)[['ax', 'ay', 'az']].transform(filter_func)

        # Handle NaN values
        nas = self.x[['ax', 'ay', 'az']].isna()
        nas_vector = nas.any(axis=1)
        if nas_vector.any():
            print(f"NAs found and replaced by 0. NA proportion: {nas_vector.mean()}")
            self.x.loc[nas_vector, ['ax', 'ay', 'az']] = 0

    def smooth_data(self):
        """Apply a smoothing operation to the data using a rolling standard deviation."""
        def smooth_func(col):
            return np.concatenate((np.zeros(12), pd.Series(col).rolling(window=25, min_periods=1).std().fillna(0).values, np.zeros(12)))

        self.x[['ax', 'ay', 'az']] = self.x.groupby(self.x2)[['ax', 'ay', 'az']].transform(smooth_func)

    def apply_kmeans(self):
        """Apply KMeans clustering to classify data points."""
        def kmeans_func(group):
            km = KMeans(n_clusters=2)
            km.fit(group)
            high_state = np.argmax(km.cluster_centers_)
            return (km.labels_ == high_state).astype(float)
        
        if self.timeDays != 0:
            self.x[['ax', 'ay', 'az']] = self.x.groupby('timeBin')[['ax', 'ay', 'az']].transform(kmeans_func)
        else:
            self.x[['ax', 'ay', 'az']] = self.x[['ax', 'ay', 'az']].transform(kmeans_func)

    def combine_data(self):
        """Combine the processed data into a final result."""
        self.x['pca'] = self.x[['ax', 'ay', 'az']].prod(axis=1)
        result = self.x[['time', 'pca']].groupby('time').max().reset_index()
        return result

    def process(self):
        """Run the full process and return the final result."""
        self.preprocess_time()
        self.apply_high_pass_filter()
        self.smooth_data()
        self.apply_kmeans()
        return self.combine_data()

# Example usage
data = pd.DataFrame({
    'time': pd.date_range(start='2023-01-01', periods=100, freq='S'),
    'ax': np.random.randn(100),
    'ay': np.random.randn(100),
    'az': np.random.randn(100),
    'segmentID': np.repeat([1, 2, 3, 4], 25)
})

pcab = PreyCatchAttemptBehaviours(x=data, x2=data['segmentID'])
result = pcab.process()
print(result)

