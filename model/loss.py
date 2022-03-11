import torch
from torch import nn
from torchaudio.transforms import Spectrogram


class TFLoss(object):

    def __init__(self, segment_window=512, segment_stride=256, alpha=0.2):
        self.alpha = alpha
        self.spectrogram = Spectrogram(n_fft=segment_window,
                                       hop_length=segment_window-segment_stride,
                                       window_fn=torch.hamming_window,
                                       center=False,
                                       power=None,
                                       onesided=False)
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.mse_loss = nn.MSELoss(reduction='mean')

    def __call__(self, output, target):
        spec_ri_mag_output = torch.abs(torch.view_as_real(self.spectrogram(output)))
        spec_ri_mag_target = torch.abs(torch.view_as_real(self.spectrogram(target)))
        loss_F = self.l1_loss(torch.sum(spec_ri_mag_output, dim=-1, keepdim=False), torch.sum(spec_ri_mag_target, dim=-1, keepdim=False))
        loss_T = self.mse_loss(output, target)

        return self.alpha * loss_T + (1-self.alpha) * loss_F
