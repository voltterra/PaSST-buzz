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

from torch.utils.data import Dataset, IterableDataset


SAMPLING_RATE=32000


DataPoint = namedtuple('DataPoint', ['path', 'label', 'name'])


def find_final_dataset_path(search_dir, search_tag):
    subdirs = os.listdir(os.path.join(search_dir, search_tag))
    subdirs = list(filter(lambda x: re.match(r'^\d+', x), subdirs))
    final_subdir = sorted(subdirs, key=lambda x: int(re.sub('\D', '', x)))[-1]
    
    return os.path.join(search_dir, search_tag, final_subdir)

       
class BuzzSplittedDatapoint():
    def __init__(self, base_name, variant_name, signal_dir, label, window_len=10, window_hop=7, SR=SAMPLING_RATE):
        self.base_name = base_name
        self.variant_name = variant_name
        self.label = label
        self.signal_dir = signal_dir
        self.window_len = 10
        self.window_hop = 7
        self.SR = SR 
        
        self._iterator = None
        
    def __next__(self):
        variant_basename, ext = os.path.splitext(self.variant_name)
        i, windowed_signal = next(self._iterator)
        
        return self.base_name, f"{variant_basename}_{i}{ext}", windowed_signal, self.label
        
    def __iter__(self):
        _load_path = os.path.join(self.signal_dir, self.variant_name)
        source_signal, _ = torchaudio.load(_load_path)
        self._iterator = enumerate(split_signal_overlapping_windows(source_signal, self.window_len, self.window_hop, self.SR), 1)
        
        return self
    
    
class BuzzIterableDataset(IterableDataset):
    """Iterable to split original files into 10sec windows"""
    def __init__(
        self, 
        signal_dir, 
        total_classes, 
        metadata_csv='metadata.csv', 
        shuffle=True, 
        subsample=1.0, 
        clip_len=10, 
        window_hop=7,
        SR=SAMPLING_RATE,
        debug=False
    ):
        self._start_df = pd.read_csv(metadata_csv)
        self.total_classes = total_classes
        
        self.signal_dir = signal_dir
        self.SR = SR
        self.clip_len = clip_len
        self.window_hop = window_hop
        
        self.start = 0
        self.end = self._start_df.shape[0] - 1
        
        self.shuffle = shuffle
        self.subsample = subsample
        self._debug = debug
        
        self._iterator = None
        
    def __next__(self):
        return next(self._iterator)
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
        else:
            np_seed = torch.initial_seed() % (2**32 - 1)            
            np.random.seed(np_seed)
            
            if (self._debug):
                print(f"Reseeding numpy on worker {worker_info.id} with a derived seed {np_seed}")            
                print(f"Worker id {worker_info.id}; torch.seed {torch.initial_seed()}")
                             
            worker_chunk = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * worker_chunk
            iter_end = min(iter_start + worker_chunk, self.end)
            
        self._iterator = self._get_iter_over_samples(self._get_iter_over_datapoints(iter_start, iter_end)) 
        return self
    
    def _shuffle_subsample_indices(self, start, end):
        loc_index = np.arange(start, end)
        size = int(np.floor(len(loc_index) * self.subsample))
        if self.shuffle:
            loc_index = np.random.choice(loc_index, size=size, replace=False)
        else:
            loc_index = loc_index[:size]
            
        return loc_index
    
    def _construct_iter_over_datapoints(self, loc_index):        
        def _gen(indexer):
            data_iterator = self._start_df.iloc[indexer, :].itertuples(index=False)                
            for original_fname, label in data_iterator:
                yield BuzzSplittedDatapoint(
                    original_fname,
                    original_fname,
                    self.signal_dir,
                    label
                )         
                
        return _gen(loc_index)

    def _get_iter_over_datapoints(self, start, end):
        loc_index = self._shuffle_subsample_indices(start, end)  
        return self._construct_iter_over_datapoints(loc_index)
            
    def _get_iter_over_samples(self, data_points_iter):
        for dp in data_points_iter:
            for basename, variant_basename, windowed_signal, label in dp:
                ohe_label = torch.nn.functional.one_hot(torch.tensor(label), num_classes=self.total_classes).type(torch.FloatTensor)
                windowed_signal = torch.Tensor(windowed_signal).reshape(-1)
                yield basename, variant_basename, windowed_signal, ohe_label
    
    def _get_random_sample_for_mixup(self):
        rand_point_idx = np.random.randint(0, self._start_df.shape[0])
        # sample one sample, make an iter out of it, and get one of the windowed_signal out of it
        rand_point_splitted = list(self._get_iter_over_samples(self._construct_iter_over_datapoints([rand_point_idx])))
        rand_chunk = np.random.randint(0, len(rand_point_splitted))
        
        return rand_point_splitted[rand_chunk]
                
                
class BuzzAugmentedIterableDataset(BuzzIterableDataset):
    """Iterator over augmented signals"""
    def __init__(self, signal_dir, total_classes, metadata_csv='metadata.csv', shuffle=True, subsample=1.0, clip_len=10, window_hop=7, SR=SAMPLING_RATE):
        super().__init__(signal_dir, total_classes, metadata_csv, shuffle, subsample, clip_len, window_hop, SR)
        self._init_start_df()
        
    def _init_start_df(self):
        self._start_df["file_name"] = self._start_df["file_name"].map(lambda x: x.split(", "))
        
    def _get_iter_over_datapoints(self, start, end):
        loc_index = self._shuffle_subsample_indices(start, end)        
        
        for original_fname, augmented_fnames, label in self._start_df.iloc[loc_index, :].itertuples(index=False):
            augmented_fname = np.random.choice(augmented_fnames, replace=False)
            yield BuzzSplittedDatapoint(
                original_fname,
                augmented_fname,
                self.signal_dir,
                label
            )