import numpy as np

from torch.optim import lr_scheduler


def exp_rampup(rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    def wrapper(epoch):
        if epoch < rampup_length:
            epoch = np.clip(epoch, 0.5, rampup_length)
            phase = 1.0 - epoch / rampup_length
            return float(np.exp(-5.0 * phase * phase))
        else:
            return 1.0
        
    return wrapper


def linear_rampdown(rampdown_length, start=0, last_value=0):
    """Linear rampup -(start)- (rampdown_length) \ _(for the rest)  """
    def wrapper(epoch):
        if epoch <= start:
            return 1.
        elif epoch - start < rampdown_length:
            return last_value + (1. - last_value) * (rampdown_length - epoch + start) / rampdown_length
        else:
            return last_value
    return wrapper


def exp_warmup_linear_down(warmup, rampdown_length, start_rampdown, last_value):
    rampup = exp_rampup(warmup)
    rampdown = linear_rampdown(rampdown_length, start_rampdown, last_value)
    
    def wrapper(epoch):
        return rampup(epoch) * rampdown(epoch)
    
    return wrapper


def get_scheduler_lambda(warm_up_len=5, ramp_down_start=20, ramp_down_len=20, last_lr_value=0.01,
                         schedule_mode="exp_lin"):
    
    if schedule_mode == "exp_lin":
        return exp_warmup_linear_down(warm_up_len, ramp_down_len, ramp_down_start, last_lr_value)
    
    
class ReduceLROnPlateauHumanized(lr_scheduler.ReduceLROnPlateau):      
    def get_last_lr(self):
        if hasattr(self, '_last_lr'):
            return self._last_lr[0]
        else:
            return self.optimizer.param_groups[0]['lr']