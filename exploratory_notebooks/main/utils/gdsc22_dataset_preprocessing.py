import math
import os
import datetime
import re
import shutil
import pathlib
import multiprocessing

from collections import namedtuple

import numpy as np
import torch
import torchaudio
import pandas as pd
import tqdm
import scipy.signal


def fuzzy_begining_overlapping_windows(signal, window_len, total_windows, SR):
    """ Clever way to increase total number of samples by starting the short signal from different points"""
    
    
    signal_original_len = signal.shape[1]
    # min of the half of the original len, or fuzzy_interval
    _interval = int(np.floor(signal_original_len / 2))
    
    rand_begins = []
    for array in np.array_split(np.arange(_interval, dtype=int), total_windows):
        rand_begins.append(np.random.choice(array))
       
    # padding to window_len:
    fuzzied_signals = []
    for begin in rand_begins:
        end = begin + window_len * SR
        _signal = signal[:, begin:end]
        pad_size = window_len * SR - _signal.shape[1]
            
        #pad_size = window_len * SR - signal_original_len + begin
        fuzzied_signals.append(np.pad(_signal, ((0, 0), (0, pad_size))))
    
    return fuzzied_signals
    

def split_signal_overlapping_windows(signal, window_len, window_hop, SR):
    """ Automatically pad and split signal into overlapping windows of size window_len.
    The overlap is determined by the formula `window_len - window_hop`
    
    NOTE: Doesn't work with fractional window_len and window_hop - need to think how to make it work
    """
    signal_original_len = signal.shape[1]  # assuming 1st dim is channel
    
    if (signal_original_len - window_len * SR) < 0:
        pad_size = window_len * SR - signal_original_len
    else:
        # if signal is divisible a whole, remove unnessesary 1-pad
        remainder = int(bool((signal_original_len - window_len * SR) % (window_hop * SR)))
        
        pad_size = remainder * (window_hop * SR) - (signal_original_len - window_len * SR) % (window_hop * SR)
    signal = np.pad(signal, ((0, 0), (0, pad_size)))
    
    window_count = (signal.shape[1] - window_len * SR) / (window_hop * SR) + 1
    windows_idx = np.arange(SR * window_len, dtype=int).reshape(1, -1) + window_hop * SR * np.arange(window_count, dtype=int).reshape(-1, 1)
    signal = np.squeeze(signal[:, windows_idx], axis=0)
    
    return np.split(signal, window_count, axis=0)


class BuzzDatapointException(Exception):
    pass


class BuzzDatapointDryRunException(BuzzDatapointException):
    pass


class BuzzDatapointNoPairedPeaksFound(BuzzDatapointException):
    pass


class BuzzOriginalSplittedDatapoint():
    def __init__(
        self, 
        base_name, 
        signal_dir, 
        label,
        *,
        train_dataset,
        short_only=None,
        long_only=None,
        window_len=10, 
        window_hop=10,
        split_to_win_min_len=30,
        fuzzy_begin_count_windows=10,
        peak_prominence=0.01, 
        peak_width=0.0001,
        silence_theshold=0.003,
        debug=False,
        dry_run=False
    ):
        self.base_name = base_name
        self.signal_dir = signal_dir
        self.label = label
        self.split_to_win_min_len = split_to_win_min_len
        self.window_len = window_len
        self.window_hop = window_hop
        
        self.fuzzy_begin_count_windows = fuzzy_begin_count_windows

        self.prominence = peak_prominence
        self.width = peak_width           #in fractions of seconds (milliseconds)       
        self.silence_theshold = silence_theshold
               
        self._iterator = None
        self.SR = None
        
        self.train_dataset = train_dataset
        
        self.short_only = short_only
        self.long_only = long_only
                      
        if all(isinstance(elem, bool) for elem in [short_only, long_only]) and (short_only + long_only) == 2:
            raise BuzzDatapointException("Mutually exclusive `short_only` and `long_only`")
            
        self._debug = debug
        
        if dry_run:
            print(locals())            
            raise BuzzDatapointDryRunException(None)
        
    def __next__(self):
        basename, ext = os.path.splitext(self.base_name)
        i, windowed_signal = next(self._iterator)
        
        return self.base_name, f"{basename}_{i}{ext}", windowed_signal, self.label, self.SR
        
    def __iter__(self):
        _load_path = os.path.join(self.signal_dir, self.base_name)
        source_signal, SR = torchaudio.load(_load_path)
        
        self.SR = SR
        
        if self.train_dataset:
            source_signal = self._remove_trailing_leading_silence(source_signal)
            
        if self.train_dataset and source_signal.shape[1] / SR >= self.split_to_win_min_len:
            self.log_debug(f"{self.base_name} Long sample ({source_signal.shape[1] / SR}) and normal splitting")
            
            if self.short_only == True:
                self._iterator = enumerate([], 1)
                self.log_debug(f"{self.base_name} skipped as too long")
            
                return self
        
            splitted_signal = split_signal_overlapping_windows(source_signal, self.window_len, self.window_hop, self.SR)
        elif not self.train_dataset:
            splitted_signal = split_signal_overlapping_windows(source_signal, self.window_len, self.window_hop, self.SR)
        elif self.train_dataset and source_signal.shape[1] / SR < self.split_to_win_min_len:
            
            if self.long_only == True:
                self._iterator = enumerate([], 1)
                self.log_debug(f"{self.base_name} skipped as too short")
                
                return self
            
            self.log_debug(f"{self.base_name} Sample size ({source_signal.shape[1] / SR}) and fuzzy begining splitting")            
            splitted_signal = fuzzy_begining_overlapping_windows(source_signal, self.window_len, self.fuzzy_begin_count_windows, self.SR)
        
        before_filtering = len(splitted_signal)
        if self.train_dataset:        
            splitted_signal = self._filter_silent_windows(splitted_signal, threshold=self.silence_theshold)
        after_filtering = len(splitted_signal)
        
        if before_filtering > after_filtering:
            self.log_debug(f"'{self.base_name}': before filtering {before_filtering}/after filtering {after_filtering} windows")
            
        self._iterator = enumerate(splitted_signal, 1)
        
        return self
    
    def log_debug(self, msg):
        if self._debug:
            print(msg)
    
    def _remove_trailing_leading_silence(self, signal):
        signal_len = signal.shape[1] / self.SR
        if (signal_len < self.window_len):
            self.log_debug(f"No silence remove applied: File '{self.base_name}' is too short ({signal_len}sec)")            
            return signal
        
        mean = float(signal.mean())
        peaks_found = scipy.signal.find_peaks(signal.reshape(-1), height=mean, prominence=self.prominence, width=self.SR * self.width)
        # Can raise, if signal is completely emtpy. Need to FIX it when breaks :D
        try:
            if len(peaks_found[0]) == 1:
                raise BuzzDatapointNoPairedPeaksFound(f"Couldn't find two consequtive peaks in '{self.base_name}'")    
            signal_left, signal_right = peaks_found[0][0], peaks_found[0][-1]
            
        except IndexError:
            self.log_debug(f"Remove silence failed: An entire file '{self.base_name}' would be removed, skipping...")
            signal_left = 0            
            signal_right = signal.shape[1]
            
        except NoPairedPeaksFound as exp:            
            self.log_debug(f"Remove silence failed: {exp}, skipping...")
            signal_left = 0            
            signal_right = signal.shape[1]            
        
        return signal[:, signal_left:signal_right]
    
    def _filter_silent_windows(self, splitted_signal, threshold):
        # Courtesy of GSK
        def test_window_empty(win_num, window):
            signal_range = window.max(axis=1) - window.min(axis=1)
            if signal_range < threshold:
                np.set_printoptions(precision=13)                
                print(window)
                self.log_debug(f"{self.base_name}: Window '{win_num}' didn't pass Threshold '{threshold}'; used SignalRange '{signal_range}'")
            return signal_range > threshold
               
        passed_windows = []
        for win_i, window in enumerate(splitted_signal, 1):
            if test_window_empty(win_i, window):
                passed_windows.append(window)
        
        return passed_windows            
   

class BuzzOriginalSplitDatasetParallel():
    def __init__(self, signal_dir, target_dir, *, dp_class_args, MODEL_SR, train_dataset=True, metadata_csv='metadata.csv', debug=False):
        """
        dp_class_args = {
            'window_len': 10,
            'window_hop': 7,
        }
        """
        self._start_df = pd.read_csv(metadata_csv)
        
        self.signal_dir = signal_dir
        self.target_dir = target_dir
        self.train_dataset = train_dataset
        
        self.MODEL_SR = MODEL_SR
        
        self.dp_class_args = dp_class_args
        
        self._debug = debug
        if debug:
            self.dp_class_args.update({
                'debug': debug,
            })
            
        self.dp_class_args.update({
            'train_dataset': train_dataset,
        })            
                                    
    def worker_process_datapoint(self, dp: BuzzOriginalSplittedDatapoint):
        # Avoid excessive concurrency, that might slow down reads.
        torch.set_num_threads(1)
        
        materialized_datapoints = []
        for basename, basename_split, windowed_signal, label, sampling_rate in dp:    
            if sampling_rate != self.MODEL_SR:
                windowed_signal = torchaudio.functional.resample(torch.Tensor(windowed_signal), sampling_rate, self.MODEL_SR)
                            
            materialized_path = os.path.join(self.target_dir, basename_split)
            torchaudio.save(materialized_path, windowed_signal, self.MODEL_SR)
            
            materialized_datapoints.append((basename, basename_split, label))
            
        return materialized_datapoints
    
    def _run_main_train(self):
        """
        Split data into 10sec windows with overlap and create a new dataframe with splitted chunks.
        Apply resampling if needed and propagate windowing parameters as required.
        """
        
        print(f"Generating new train/val files in '{self.target_dir}' with splitted data...")        
        
        data_points = []
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            for dps in tqdm.tqdm(pool.imap_unordered(self.worker_process_datapoint, self._get_iter_over_train_datapoints()), total=self._start_df.shape[0]):
                data_points.extend(dps)
            
        df = pd.DataFrame(data_points, columns=["original_fname", "name_split", "label"])
        df.to_csv(os.path.join(self.target_dir, 'metadata.csv'), index=False)
    
    def _run_main_test(self):
        print(f"Generating new test files in '{self.target_dir}' with splitted data...")        
        
        data_points = []
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            for dps in tqdm.tqdm(pool.imap_unordered(self.worker_process_datapoint, self._get_iter_over_test_datapoints()), total=self._start_df.shape[0]):
                data_points.extend(dps)
            
        df = pd.DataFrame(data_points, columns=["original_fname", "name_split", "__REMOVE__"])
        df = df.drop("__REMOVE__", axis=1)
        df.to_csv(os.path.join(self.target_dir, 'metadata.csv'), index=False)
          
    def run_main(self):
        self.prepare_target_dir()
        
        if self.train_dataset:
            self._run_main_train()
        else:
            self._run_main_test()
                
    def prepare_target_dir(self):          
        os.makedirs(self.target_dir, exist_ok=False)        
        
    def _get_iter_over_train_datapoints(self):
        for original_fname, label in self._start_df.itertuples(index=False):
            
            yield BuzzOriginalSplittedDatapoint(
                original_fname,
                self.signal_dir,
                label,
                **self.dp_class_args
            )
            
    def _get_iter_over_test_datapoints(self):
        for original_fname in self._start_df.itertuples(index=False):
            original_fname = original_fname[0]
            
            yield BuzzOriginalSplittedDatapoint(
                original_fname,
                self.signal_dir,
                "!TEST!",
                **self.dp_class_args
            )   