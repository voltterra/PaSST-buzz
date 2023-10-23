import functools

import torch
import torchaudio
import numpy as np

from torch.utils.data import IterableDataset, Dataset



class AbstractTransformation():
    def __init__(self, debug=False):
        self._debug = debug
        
    def log_debug(self, msg):
        if self._debug:
            print(msg)
        
    def _set_debug(self, value):
        self._debug = bool(value)
        

class BaseDatasetTransformer(AbstractTransformation):
    pass
        

class BaseSpectrogramTransformer(AbstractTransformation):
    pass


class AmplitudeScaler():
    def __init__(self, target_dBFS=-20):
        """Only works with float32 torch tensors"""
        
        self.target_dBFS = target_dBFS         
        
    def __call__(self, signal):
        """Return a normalized signal and delta_dBFS calculated
        """
        # Flat tensor, 
        if len(signal.shape) == 1:
            _dim = 0
        elif len(signal.shape) == 2:
        # (CH, SAMPLES)
            _dim=1
            
        # Could have avoided torch.sqrt, but then would multiply like that `10*torch.log10(rms_squared)`
        rms = torch.sqrt((signal**2).mean(dim=_dim))
        
        source_dBFS = (20 * torch.log10(rms))
        delta_dBFS = self.target_dBFS - source_dBFS
        
        #gain in real values
        gain = 10 ** (delta_dBFS / 20)
        signal = signal * gain
        
        return signal, source_dBFS, delta_dBFS


class MixUp1Level(BaseDatasetTransformer):
    """Mixup on the signal level"""
    def __init__(self, rate=0.5, alpha=0.5, beta=2, debug=False):
        self.rate = rate
        self.alpha = alpha
        self.beta = beta
        
        self.distr = torch.distributions.beta.Beta(self.alpha, self.beta)
        
        super().__init__(debug)
        
    def __call__(self, x1, y1, split_name1, x2, y2, split_name2):
        rnd = torch.rand(1)
        
        signature = f"{__class__}: Mixup yes={rnd<self.rate} "
        
        if rnd < self.rate:            
            lmb = self.distr.sample()
            lmb = max(lmb, 1. - lmb)
            x1 = x1 - x1.mean()
            x2 = x2 - x2.mean()
            x = (x1 * lmb + x2 * (1. - lmb))
            x = x - x.mean()
            
            signature += f"mixed up signals(1={split_name1}; 2={split_name2} lambda={lmb}"
            
            return x, (y1 * lmb + y2 * (1. - lmb)), signature
        
        self.log_debug(f"{__class__}: Mixing up yes=False")
        
        return x1, y1, signature


class RandomGain(BaseDatasetTransformer):
    """ Randomly increase, decrease the amp of a wave"""
    def __init__(self, max_gain=7, debug=False):
        self.max_gain = max_gain
        
        super().__init__(debug)
        
    def __call__(self, x1, y1):
        gain = torch.randint(self.max_gain * 2, (1,)).item() - self.max_gain
        amp = 10 ** (gain / 20)
        x1 = x1 * amp
        
        signature = f"{__class__}: Random gain='{amp}'"
        self.log_debug(f"{__class__}: Random gain in amp signal '{amp}'")                
        
        return x1, y1, signature
    

class Rolling(BaseDatasetTransformer):
    def __init__(self, shift_sec, SR, debug=False):
        self.shift_sec = shift_sec
        self.SR = SR
    
        super().__init__(debug)
        
    def __call__(self, x1, y1):
        low_ = int(np.ceil(self.shift_sec*self.SR))
        high_ = low_ + 1
        shift_samples = torch.randint(low=-low_, high=high_, size=(1,)).item()
        x1 = x1.roll(shift_samples)
        
        signature = f"{__class__}: Rolling shift='{shift_samples / self.SR}'"
        self.log_debug(f"{__class__}: Rolling with interval '{shift_samples / self.SR}'")                
        
        return x1, y1, signature
    
    
def power_spectral_density(psd_function):
    """
    Returns a function noise_psd with proper psd_function defined.
    Used as decorator. psd - is power spectral density. 
    """
    @functools.wraps(psd_function)
    def colored_generator(self, size):
        return self.noise_psd(size, psd_function)
            
    return colored_generator

            
class ColoredNoise(BaseDatasetTransformer):
    def __init__(self, low_SNR=15, high_SNR=25, SR=32000, debug=False):
        self.SR = SR
        self.low_SNR = low_SNR
        self.high_SNR = high_SNR
        
        self._noises = [self.white_noise, self.blue_noise, self.violet_noise, self.pink_noise, self.brownian_noise]
        
        super().__init__(debug)
        
    @staticmethod
    def add_noise(x1, noise, SNR):            
        x1_l2 = (x1**2).sum()
        noise_l2 = (noise**2).sum()
  
#         Using rms is a bit more intuitive, but the results match
#         x1_l2_rms = (x1 ** 2).mean()
#         noise_rms = (noise ** 2).mean()
        
#         print(f"x1_l2_sqrt: {torch.sqrt(x1_l2)}")
#         print(f"noise_l2_sqrt:  {torch.sqrt(noise_l2)}")
        
#         print(f"x1_rms {torch.sqrt(x1_l2_rms)}")
#         print(f"noise_rms {torch.sqrt(noise_rms)}")        
        
        original_snr = 10 * (torch.log10(x1_l2) - torch.log10(noise_l2))
        scaled_noise_dBFS = (original_snr - SNR)
        scale_snr = 10 ** (scaled_noise_dBFS / 20)
               
        return (x1 + scale_snr*noise), original_snr, scaled_noise_dBFS

    def __call__(self, x1, y1):
        size = x1.shape[0]
        
        noise_index = torch.randint(low=0, high=len(self._noises), size=(1, )).item()
        SNR = torch.randint(low=self.low_SNR, high=self.high_SNR, size=(1, )).item()
        
        fn = self._noises[noise_index]
        noise_signal = fn(size)
               
        x1, original_snr, scaled_noise_dBFS = self.add_noise(x1, noise_signal, SNR)
        
        signature = f"{__class__}: noise fun='{fn}'; snr='{SNR}'; original_SNR='{original_snr}', scaled_noise_dBFS={scaled_noise_dBFS}"
        self.log_debug(signature)   
        
        return x1, y1, signature

    def noise_psd(self, signal_size, psd_function):
        """Default is white"""
        X_white_fft = torch.fft.rfft(torch.randn(signal_size))
        # Spectral energy distr
        # See https://pytorch.org/docs/stable/generated/torch.fft.rfftfreq.html on why multiplication
        S = psd_function(self, torch.fft.rfftfreq(signal_size) * self.SR)
        # Normalize S
        S = S / torch.sqrt(torch.mean(S**2))
        X_shaped = X_white_fft * S;
        return torch.fft.irfft(X_shaped);
    
    @power_spectral_density
    def white_noise(self, freq: torch.Tensor):
        """1"""
        return torch.ones(1)

    @power_spectral_density
    def blue_noise(self, freq: torch.Tensor):
        """2"""        
        return torch.sqrt(freq)
    
    @power_spectral_density
    def violet_noise(self, freq: torch.Tensor):
        """3"""                
        return freq
    
    @power_spectral_density
    def pink_noise(self, freq: torch.Tensor):
        """4"""                        
        return 1/torch.where(freq == 0, float('inf'), torch.sqrt(freq))
    
    @power_spectral_density
    def brownian_noise(self, freq: torch.Tensor):
        """5"""                        
        return 1/torch.where(freq == 0, float('inf'), freq)
    
    
class NormalizeAmplitudes(BaseDatasetTransformer):
    def __init__(self, amp_scaler_class_args={}, debug=False):
        self.AmpScaler = AmplitudeScaler(**amp_scaler_class_args)
        
        super().__init__(debug)
        
    def __call__(self, x1, y1):
        x1, source_dBFS, change_in_dBFS = self.AmpScaler(x1)
        signature = f"{__class__}: Normalized amplitude '{source_dBFS}' by '{change_in_dBFS}'. Target dBFS='{self.AmpScaler.target_dBFS}'"
        self.log_debug(signature)
        
        return x1, y1, signature
        

class MixUp2Level(BaseSpectrogramTransformer):
    """Mixup on the Spectrogram level"""    
    # We would have to call it on the batch after model.mel input
    # As it is applied on spectrograms, not on a signal itself.
    def __init__(self, alpha=0.3, beta=0.3, debug=False):
        self.alpha = alpha
        self.beta = beta
              
        self.distr = torch.distributions.beta.Beta(self.alpha, self.beta)
        
        super().__init__(debug)
    
    def __call__(self, X_batch, y, device):
        batch_size = X_batch.shape[0]
        
        rnd_indicies = torch.randperm(batch_size)
        lmb = self.distr.sample(sample_shape=(batch_size, 1))
        lmb, _ = torch.cat([lmb, 1-lmb], dim=1).max(dim=1)
        
        self.log_debug(f"{__class__}: 2LvlMixup (1st part) lambda={lmb}")
        
        # Do the mixup on an entire batch
        lmb = lmb.to(device)
        X_batch = X_batch * lmb.reshape(batch_size, 1, 1, 1) + X_batch[rnd_indicies] * (1. - lmb.reshape(batch_size, 1, 1, 1))        
        
        # Magic - use python's coroutines to "yield back" to the code called this class to get the forward pass on augmented data, then jump back where we left off
        # going to use yield from
        cross_entropy_fn, y_hat = (yield X_batch)
              
        y_mix = y * lmb.reshape(batch_size, 1) + y[rnd_indicies] * (1. - lmb.reshape(batch_size, 1))               
        samples_loss = cross_entropy_fn(y_hat, y_mix, reduction="none")
        
        self.log_debug(f"{__class__}: 2LvlMixup (2nd part) y_mix={y_mix.cpu()}")
        self.log_debug(f"{__class__}: 2LvlMixup samples_loss={samples_loss.cpu()}")
        
        return samples_loss, y_mix
        

class BuzzIterTransformedDataset(IterableDataset):
    def __init__(self, transformers, buzz_iterable):

        self.transformers = transformers
        self._internal_iter = buzz_iterable
        
        self._iter = None
        
    def __iter__(self):
        self._iter = iter(self._internal_iter)
        return self
        
    def __next__(self):
        # Mixup needs a special treatment, as it does a separate lookup into samples
        original_fname, augmented_fname, x1, y1 = next(self._iter)
        
        for i, transformer in enumerate(self.transformers):  
            if isinstance(transformer, MixUp1Level):
                _, _, x2, y2 = self._internal_iter._get_random_sample_for_mixup()
                x1, y1 = transformer(x1, y1, x2, y2)
            else:
                x1, y1 = transformer(x1, y1)

        return (original_fname, augmented_fname, x1, y1)
    
    
class BuzzMapTransformedDataset(Dataset):
    def __init__(self, transformers, buzz_mappable, debug=False):
        self.transformers = transformers
        self._mapper = buzz_mappable
        self._debug = debug
        
        # Set the mapper for debug mode as well...
        if self._debug:
            self._mapper._debug = self._debug
            
    @property
    def signal_labels(self):
        return self._mapper.signal_labels
        
    def _init_multiprocess_worker(self, worker_id):
        if self._debug:
            print(f"Calling init for a worker {worker_id}")            
        self._mapper._init_multiprocess_worker()
        
    def __str__(self):
        _str = ", ".join([str(tr) for tr in self.transformers])
        _str_mapper = str(self._mapper)
        
        return f"{__class__}: Transformations=({_str}); Mapper=({_str_mapper})"            
        
    def __len__(self):
        return len(self._mapper)
    
    def _item_from_mapper(self, idx):
        return self._mapper[idx]
    
    def __getitem__(self, idx):
        original_name, splitted_name, x1, y1 = self._item_from_mapper(idx)
        processing_signatures = []
        
        for i, transformer in enumerate(self.transformers):
            # Mixup needs a special treatment, as it does a separate lookup into samples            
            if isinstance(transformer, MixUp1Level):
                mixup = transformer
                rnd_idx = torch.randint(len(self), size=(1,)).item()
                _, splitted_name2, x2, y2 = self._mapper[rnd_idx]
                x1, y1, signature = mixup(x1, y1, splitted_name, x2, y2, splitted_name2)
                
                # Aggregate all applied transformations
                processing_signatures.append(signature)
            else:
                x1, y1, signature = transformer(x1, y1)
                
                # Aggregate all applied transformations
                processing_signatures.append(signature)
                
        signature = ";".join(f"STEP{step}:{sig}" for step, sig in enumerate(processing_signatures, 1))
        del processing_signatures
        
        return (original_name, splitted_name, signature, x1, y1)

    
class BuzzMapTransformedTestDataset(BuzzMapTransformedDataset):
    """
    GSK: Overloading the __getitem__ to dummy y1
    DA: Overloading special method _item_from_mapper and remove code duplication :)
    """
    
    def _item_from_mapper(self, idx):
        original_name, splitted_name, x1 = self._mapper[idx]
        
        return original_name, splitted_name, x1, -1
    
    
class BuzzTorchDataLoader(torch.utils.data.DataLoader):
    def __str__(self):
        return str(self.dataset)