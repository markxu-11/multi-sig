"""
Author: Mark Xu

"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import List, Tuple, Callable
from scipy.signal import butter, filtfilt, iirnotch
import numpy as np


class PreprocessTask(ABC):
    @abstractmethod
    def setup(self) -> None:
        pass

    @abstractmethod
    def process(self, data: np.ndarray):
        pass


class Windower(PreprocessTask):
    def __init__(self, data_len: int, win_len: int, overlap: int) -> None:
        """Applies a windowing scheme to the multichannel signal.

        Args:
            data_len (int): Length of the signal in data points.
            win_len (int): Length of the window in data points.
            overlap (int): Length of the overlap between windows in data points.
        """
        self.data_len = data_len
        self.win_len = win_len
        self.overlap = overlap
        self.windows = None
    
    def setup(self):
        """Generates windowing indices
        """
        if self.data_len <= 0 or self.win_len > self.data_len or self.overlap >= self.win_len:
            raise ValueError("Invalid input arguments for Windower")
        step = self.win_len - self.overlap
        start = np.arange(0, self.data_len - self.win_len + 1, step)
        end = start + self.win_len
        self.windows = np.column_stack((start, end))
    
    def process(self, data: np.ndarray) -> np.ndarray:
        """Windows the input data.

        Args:
            data (np.ndarray): Signal data in the shape of (channel, signal)


        Returns:
            np.ndarray: Windowed signal data
        """
        if len(data.shape) > 2:
            raise ValueError(f"Dimensions exceeded, got {len(data.shape)}")
        elif len(data.shape) == 1:
            data = data[np.newaxis, :]
            
        num_channel = data.shape[0]
        windowed = np.zeros((num_channel, self.windows.shape[0], self.win_len))
        for ch in range(num_channel):
            for i, w in enumerate(self.windows):
                windowed[ch, i] = data[ch, w[0] : w[1]]
        
        return np.squeeze(windowed)


class SignalFilter(PreprocessTask):
    def __init__(self, fs: int):
        """Applies digital filtration on the signal.

        Args:
            fs (int): Sampling frequency of the signal.
        """
        self.fs = fs
        self.filters = []

    def setup(self) -> None:
        if self.fs <= 0:
            raise ValueError("Sampling frequency needs to be a positive value")
        
        if len(self.filters) == 0:
            print("No filters has been added")
    
    def add_lowpass(self, cutoff: float, order: int=5) -> None:
        b, a = butter(order, cutoff / (0.5 * self.fs), btype='low')
        self.add_filter((b, a))
    
    def add_highpass(self, cutoff: float, order: int=5) -> None:
        b, a = butter(order, cutoff / (0.5 * self.fs), btype='high')
        self.add_filter((b, a))
    
    def add_bandpass(self, lowcut: float, highcut: float, order: int=5) -> None:
        b, a = butter(order, 
                      [lowcut / (0.5 * self.fs), highcut / (0.5 * self.fs)], 
                      btype='band')
        self.add_filter((b, a))
    
    def add_notch(self, notch_freq: float, Q: float=30) -> None:
        b, a = iirnotch(notch_freq, Q, self.fs)
        self.add_filter((b, a))

    def add_filter(self, filter: Tuple[np.ndarray, np.ndarray]) -> None:
        """Add custom `ba` filter coefficients
        """
        self.filters.append(filter)
    
    def process(self, data: np.ndarray) -> np.ndarray:
        """Applies the list of filter on the last axis (inner most) of the data.

        Args:
            data (np.ndarray): Signal data where the last axis is the signal \
                segments

        Returns:
            np.ndarray: Filtered signal data
        """
        res = data
        for filter in self.filters:
            res = filtfilt(*filter, res)
        
        return res
    

class TDExtractor(PreprocessTask):
    def __init__(self):
        """Feature extracts the signal.
        """
        self.feature_methods: List[Callable] = []
        self.vec = []

    def add_vectorised_features(self, feature_methods):
        """Add vectorised helper functions for extracting time-domain features.

        Args:
            feature_methods: Vectorised functions that take a single ndarray as\
                parameter used to extract a feature over the last axis.
        """
        if isinstance(feature_methods, Iterable):
            self.feature_methods.extend(feature_methods)
            self.vec.extend([True] * len(feature_methods))
        else:
            self.feature_methods.append(feature_methods)
            self.vec.append(True)

    def add_features(self, feature_methods):
        """Add helper functions for extracting time-domain features.

        Args:
            feature_methods: Functions that take a single ndarray as parameter \
                used to extract a feature.
        """
        if isinstance(feature_methods, Iterable):
            self.feature_methods.extend(feature_methods)
            self.vec.extend([False] * len(feature_methods))
        else:
            self.feature_methods.append(feature_methods)
            self.vec.append(False)

    def setup(self):
        if not self.feature_methods:
            raise ValueError("No features has been added")
        
    def process(self, data: np.ndarray) -> np.ndarray:
        """Applies the feature methods for feature extraction over the last axis

        Args:
            data (np.ndarray): Signal data

        Returns:
            np.ndarray: Array of features
        """
        if data.ndim == 2:
            data = data[np.newaxis, :]

        res = np.zeros((data.shape[0], data.shape[1], len(self.feature_methods)))
        for i, method in enumerate(self.feature_methods):
            if self.vec[i]:
                res[:, :, i] = method(data)
            else:
                res[:, :, i] = np.apply_along_axis(method, axis=-1, arr=data)
        
        return np.squeeze(res)


class SignalPreprocessor:
    def __init__(self):
        """Manager class for PreprocessTask
        """
        self.tasks: List[PreprocessTask] = []

    def add_tasks(self, tasks: PreprocessTask):
        """Add tasks to the preprocessor manager. Order matters. 

        Args:
            tasks (PreprocessTask): Child class of PreprocessTask
        """
        if isinstance(tasks, Iterable):
            self.tasks.extend(tasks)
        else:
            self.tasks.append(tasks)
    
    def setup_tasks(self):
        """Setup all tasks
        """
        for task in self.tasks:
            task.setup()

    def process_tasks(self, data: np.ndarray):
        """Process all tasks by cascading output of a task to the input of the \
           next task in order.

        Args:
            data (np.ndarray): Multichannel signal data

        Returns:
            _type_: Preprocessed data
        """
        res = data
        for task in self.tasks:
            res = task.process(res)
        
        return res
    
