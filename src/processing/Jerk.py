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
