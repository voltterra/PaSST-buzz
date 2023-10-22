import torch

from hear21passt.base import AugmentMelSTFT
from hear21passt.models.passt import get_model


def get_pretrained_passt_model(**kwargs):
    mel = AugmentMelSTFT(n_mels=128, sr=32000, win_length=800, hopsize=320, n_fft=1024, freqm=48,
                         timem=192,
                         htk=False, fmin=0.0, fmax=None, norm=1, fmin_aug_range=10,
                         fmax_aug_range=2000,
                         #debug=kwargs['debug']
                        )
    kwargs["arch"] = kwargs.get("arch", "passt_s_kd_p16_128_ap486")
    
    # deepcopy and remove excessive mode
    passt_kwargs = dict(kwargs)
    passt_kwargs.pop("mode")
    #passt_kwargs.pop("debug")
    # returns a PaSST model with the defined kwargs
    net = get_model(**passt_kwargs)
    model = PaSSTGDSCIface(mel=mel, net=net, **kwargs)
    return model


class PaSSTGDSCIface(torch.nn.Module):
    def __init__(self, mel, net, mode, **kwargs):
        torch.nn.Module.__init__(self)
        self.mel = mel
        self.net = net
        self.mode = mode

    def forward(self, x):
        specs = self.mel(x)
        # FIX: change the size to 998 as the model expects
        # specs = specs[:, :, 1:-1]
        specs = specs.unsqueeze(1)        
        # (logits, embedding)
        x, features = self.net(specs)

        if self.mode == "all":
            return x, features
        elif self.mode == "embed_only":
            embed = features
        elif self.mode == "logits":
            embed = x
        else:
            raise RuntimeError(f"mode='{self.mode}' is not recognized not in: all, embed_only, logits")
            
        return embed