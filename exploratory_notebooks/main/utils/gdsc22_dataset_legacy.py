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

from torch.utils.data import Dataset, IterableDataset

from utils.gdsc22_dataset_preprocessing import split_signal_overlapping_windows
from utils.gdsc22_dataset import BuzzSplittedDatapoint


SAMPLING_RATE=32000


class BuzzDatasetLegacy(Dataset):
    def __init__(self, wav_dir, metadata_csv='metadata.csv', clip_len=10, sr=SAMPLING_RATE):
        self.wav_labels = pd.read_csv(metadata_csv)
        self.wav_dir = wav_dir
        self.sr = sr
        self.clip_len = clip_len
        
    def __len__(self):
        return self.wav_labels.shape[0]
    
    def __getitem__(self, idx):
        fname = self.wav_labels.iloc[idx, 0]
        _load_path = os.path.join(self.wav_dir, fname)
        wav, sr = torchaudio.load(_load_path)
        #get rid of channel dimension.
        wav = wav.reshape(-1)
        
        wav = self.pad_truncate(wav)
        
        # GSK added this: think/hope it doens't break anything in training
        if self.wav_labels.shape[1] == 1:
            return wav, fname
        else:
            label = self.wav_labels.iloc[idx, 1]
            return wav, label, fname
    
    def pad_truncate(self, wav):
        audio_len = self.clip_len * self.sr
        wav_len = wav.shape[0]
        
        if wav_len == audio_len:
            return wav
        
        elif wav_len < audio_len:
            return torch.cat([wav, torch.zeros(audio_len - wav_len, dtype=torch.float32)], dim=0)
        
        else:
            offset = torch.randint(0, wav_len - audio_len + 1, (1, )).item()
            return wav[offset:offset + audio_len]
          
    
class BuzzMapDataset(Dataset):
    def __init__(self, signal_dir, num_classes, *, sampling_fraction=1.0, metadata_csv='metadata.csv', sr=SAMPLING_RATE, debug=False):
        """ It doesn't do any stratification when fully sampled. The builtin strat is used only with subsampling.
        Implemented for augmentation
        """
        self.signal_labels = pd.read_csv(metadata_csv)
               
        self._metadata_path = metadata_csv
        self.signal_dir = signal_dir
        self.num_classes = num_classes
        self.SR = sr
        
        self.sampling_fraction = sampling_fraction
        
        self._debug = debug
        
        # if sampling_fraction < 1.0:
        #     raise Exception("BETTER OF USING TORCH's Weighted Random Sampler with helpers :)")
        #     self.signal_labels = self._sample_df_stratified()        
        
#     def _sample_df_stratified(self):
#         _df = self.signal_labels
        
#         sampling_per_label = _df.shape[0] * self.sampling_fraction / self.num_classes
        
#         def sampler(grouped_rows, n_per_label):
#             frac = n_per_label / len(grouped_rows)            
#             if frac > 1.0:
#                 return grouped_rows.sample(frac=frac, replace=True)
#             else:
#                 return grouped_rows.sample(frac=frac)
        
#         _df_stratified = _df.groupby(['label'], as_index=False).apply(lambda x: sampler(x, sampling_per_label)).reset_index().drop(['level_0', 'level_1'], axis=1)
#         return _df_stratified
        
        
    def __str__(self):
        #return f"{__class__}: reading data from {self._metadata_path}, sampling_fraction {sampling_fraction}. Dataset stats: shape={self.signal_labels.shape}"
        return f"{__class__}: reading data from {self._metadata_path}. Dataset stats: shape={self.signal_labels.shape}"
        
    def _init_multiprocess_worker(self):
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is not None:
            # Multiprocessing - needs a reseed
            np_seed = torch.initial_seed() % (2**32 - 1)            
            np.random.seed(np_seed)            
            if (self._debug):            
                print(f"Reseeding numpy on worker {worker_info.id} with a derived seed {np_seed}")            
                print(f"Worker id {worker_info.id}; torch.seed {torch.initial_seed()}")        
        
    def __len__(self):
        return self.signal_labels.shape[0]
    
    def __getitem__(self, idx):
        name_original, name_split, label = self.signal_labels.iloc[idx, :]
        ohe_label = torch.nn.functional.one_hot(torch.tensor(label), num_classes=self.num_classes).type(torch.FloatTensor)        
        
        _load_path = os.path.join(self.signal_dir, name_split)
        wav, SR = torchaudio.load(_load_path)
        if self.SR != SR:            
            raise ValueError(f"Input data is not properly resampled!. Expected {self.SR}, got {SR}")
            
        #get rid of channel dimension.
        wav = wav.reshape(-1)
        
        return name_original, name_split, wav, ohe_label
    
    
class BuzzDatasetWeighter():           
    def __call__(self, *args):
        dfs = [ds.signal_labels for ds in args]
        
        result_df = pd.concat(dfs)        
        weights = self._calc_weights(result_df)
        
        return weights, result_df.shape[0]
    
    def _calc_weights(self, df):
        freq = df['label'].value_counts()
        freq = 1/freq
        weight = freq.to_frame(name="weight")
        
        df = df.join(weight, on='label')
                
        return df['weight'].values
    

        
class BuzzMapTestDataset(BuzzMapDataset):
    def __getitem__(self, idx):
        
        name_original, name_split = self.signal_labels.iloc[idx, :]
        
        _load_path = os.path.join(self.signal_dir, name_split)
        wav, SR = torchaudio.load(_load_path)
        if self.SR != SR:            
            raise ValueError(f"Input data is not properly resampled!. Expected {self.SR}, got {SR}")
            
        #get rid of channel dimension.
        wav = wav.reshape(-1)
        
        return name_original, name_split, wav
    
######################################################################################################################################
#                                         LEGACY CODE MARKED FOR FUTURE REMOVAL                                                      #
######################################################################################################################################   
    
class BuzzAugmentedDataset(Dataset):
    def __init__(self, signal_dir, metadata_csv='metadata.csv', clip_len=10, window_hop=7, SR=SAMPLING_RATE, tmp_dir="/root/data/tmp"):
        self._start_df = pd.read_csv(metadata_csv)
        self._init_start_df()
        
        self.signal_dir = signal_dir
        self.SR = SR
        self.clip_len = clip_len
        self.window_hop = window_hop
        
        self._materialized_df = None
        self.tmp_dir = tmp_dir
        
    def __len__(self):
        if not self.is_materialized:
            self._materialize_df()
            
        return self._materialized_df.shape[0]
    
    def __getitem__(self, idx):
        if not self.is_materialized:
            self._materialize_df()            
        
        fname, augmented_fname, materialized_path, label = self._materialized_df.iloc[idx, [0, 1, 2, 3]]
        
        wav, _ = torchaudio.load(materialized_path)
        return wav, label, fname, augmented_fname
    
    def _init_start_df(self):
        self._start_df["file_name"] = self._start_df["file_name"].map(lambda x: x.split(", "))    
          
    def _materialize_df(self):
        """ Sample each "sample" and then pick one of the variant randomly.
        Split it into 10sec windows with overlap and join in one dataframe.
        """
        
        print("Generating new random dataframe with augmented data...")        
        self._prepare_temp_dirs()
        
        data_points = []
        for original_fname, augmented_fnames, label in self._start_df.itertuples(index=False):
            augmented_fname = np.random.choice(augmented_fnames)
            points = BuzzSplittedDatapoint(                
                original_fname,
                augmented_fname,
                self.signal_dir,
                label
            )
            for _, variant_basename, signal, _ in points:
                vb, ext = os.path.splitext(variant_basename)
                materialized_path = os.path.join(self.tmp_dir, f"{vb}.wav")
                torchaudio.save(materialized_path, torch.Tensor(signal), self.SR)
                data_points.append((original_fname, variant_basename, materialized_path, label))
            
        self._materialized_df = pd.DataFrame(data_points, columns=["original_fname", "augmented_fname", "materialized_path", "label"])        
        
    def _prepare_temp_dirs(self):
        if pathlib.Path(self.tmp_dir).exists():
            shutil.rmtree(self.tmp_dir)
            
        os.makedirs(self.tmp_dir, exist_ok=True)
        
    @property
    def is_materialized(self):
        return False if self._materialized_df is None else True


class BuzzAugmentedDatasetParallel(BuzzAugmentedDataset):
    def worker_process_datapoint(self, dp: BuzzSplittedDatapoint):
        # Avoid excessive concurrency, that might slow down reads.
        torch.set_num_threads(1)
        materialized_datapoints = []
        for basename, variant_basename, windowed_signal, label in dp:
            
            variant_name, variant_ext = os.path.splitext(variant_basename)
            materialized_path = os.path.join(self.tmp_dir, f"{variant_name}.wav")
            torchaudio.save(materialized_path, torch.Tensor(windowed_signal), self.SR)
            
            materialized_datapoints.append((basename, variant_basename, materialized_path, label))
            
        return materialized_datapoints
    
    def _materialize_df(self):
        """ Sample each "sample" and then pick one of the variant randomly.
        Split it into 10sec windows with overlap and join in one dataframe.
        """
        
        print("Generating new random dataframe with augmented data...")        
        self._prepare_temp_dirs()
        
        data_points = []
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            for dps in tqdm.tqdm(pool.imap_unordered(self.worker_process_datapoint, self._get_iter_over_datapoints()), total=self._start_df.shape[0]):
                data_points.extend(dps)
            
        self._materialized_df = pd.DataFrame(data_points, columns=["original_fname", "augmented_fname", "materialized_path", "label"]) 
        
    def _get_iter_over_datapoints(self):
        for original_fname, augmented_fnames, label in self._start_df.itertuples(index=False):
            augmented_fname = np.random.choice(augmented_fnames)
            
            yield BuzzSplittedDatapoint(
                original_fname,
                augmented_fname,
                self.signal_dir,
                label
            )
            
    def _get_iter_over_samples(self, data_points_iter):
        # Avoid excessive concurrency, that might slow down reads.
        for basename, variant_basename, windowed_signal, label in data_points_iter:          
            yield basename, variant_basename, torch.Tensor(windowed_signal), label