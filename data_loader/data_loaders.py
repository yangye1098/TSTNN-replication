from base import BaseDataLoader

import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
from math import ceil


def generate_inventory(path, file_type='.wav'):
    path = Path(path)
    assert path.is_dir(), '{:s} is not a valid directory'.format(path)

    file_paths = path.glob('*'+file_type)
    file_names = [ file_path.name for file_path in file_paths ]
    assert file_names, '{:s} has no valid {} file'.format(path, file_type)
    return file_names


class AudioDataset(Dataset):
    def __init__(self, data_root, datatype, sample_rate=8000, T=-1):
        if datatype not in ['.wav', '.spec.npy', '.mel.npy']:
            raise NotImplementedError
        self.datatype = datatype
        self.sample_rate = sample_rate
        # number of frame to load
        self.T = T

        self.clean_path = Path('{}/clean'.format(data_root))
        self.noisy_path = Path('{}/noisy'.format(data_root))

        self.inventory = generate_inventory(self.clean_path, datatype)
        self.data_len = len(self.inventory)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):

        if self.datatype == '.wav':
            clean, sr = torchaudio.load(self.clean_path/self.inventory[index])
            assert(sr == self.sample_rate)
            noisy, sr = torchaudio.load(self.noisy_path/self.inventory[index])
            assert (sr == self.sample_rate)
            n_frames = clean.shape[-1]
            assert (n_frames == noisy.shape[-1])

            if n_frames > self.T > 0:
                start_frame = torch.randint(0, n_frames - self.T, [1])
                clean = clean[:, start_frame:(start_frame+self.T)]
                noisy = noisy[:, start_frame:(start_frame+self.T)]

            elif self.T > n_frames > 0:
                clean = F.pad(clean, (0, self.T - n_frames), 'constant', 0)
                noisy = F.pad(noisy, (0, self.T - n_frames), 'constant', 0)


        elif self.datatype == '.spec.npy' or self.datatype == '.mel.npy':
            # load the two grams
            clean = torch.from_numpy(np.load(self.clean_path/self.inventory[index]))
            noisy = torch.from_numpy(np.load(self.noisy_path/self.inventory[index]))

        return clean, noisy, index

    def getName(self, idx):
        name = self.inventory[idx].rsplit('.', 1)[0]
        return name



class AudioDataLoader(BaseDataLoader):
    """
    Load Audio data
    """
    def __init__(self, dataset,  batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        self.dataset =dataset
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)



class InferDataset(AudioDataset):

    def __getitem__(self, index):

        if self.datatype == '.wav':
            clean, sr = torchaudio.load(self.clean_path/self.inventory[index])
            assert(sr == self.sample_rate)
            noisy, sr = torchaudio.load(self.noisy_path/self.inventory[index])
            assert (sr == self.sample_rate)
            n_frames = clean.shape[-1]
            assert (n_frames == noisy.shape[-1])
            start_frame = 0
            n_chunck = ceil(n_frames / self.T)

            clean = F.pad(clean, (0, n_chunck*self.T - n_frames), 'constant', 0)
            noisy = F.pad(noisy, (0, n_chunck*self.T - n_frames), 'constant', 0)

            clean_stacked = clean.view(n_chunck, 1, self.T)
            noisy_stacked = noisy.view(n_chunck, 1, self.T)

            index_tensor = index*torch.ones(n_chunck, dtype=torch.long)

        elif self.datatype == '.spec.npy' or self.datatype == '.mel.npy':
            raise NotImplementedError

        return clean_stacked, noisy_stacked, index_tensor

def infer_data_collate(batch):

    for i, (clean_stacked, noisy_stacked, index_tensor) in enumerate(batch):
        if i == 0:
            clean_collated = clean_stacked
            noisy_collated = noisy_stacked
            index_collated = index_tensor
        else:
            clean_collated = torch.cat([clean_collated, clean_stacked], dim=0)
            noisy_collated = torch.cat([noisy_collated, noisy_stacked], dim=0)
            index_collated = torch.cat([index_collated, index_tensor], dim=0)

    return clean_collated, noisy_collated, index_collated



class InferDataLoader(BaseDataLoader):
    """
    Load Audio data
    """
    def __init__(self, dataset,  batch_size, num_workers=1):
        self.dataset =dataset
        super().__init__(self.dataset, batch_size, shuffle=False, validation_split=0, num_workers=num_workers, collate_fn=infer_data_collate)


class OutputDataset(AudioDataset):
    def __init__(self, data_root, datatype, sample_rate=8000, T=-1):
        if datatype not in ['.wav', '.spec.npy', '.mel.npy']:
            raise NotImplementedError
        self.datatype = datatype
        self.sample_rate = sample_rate
        # number of frame to load
        self.T = T

        self.clean_path = Path('{}/target'.format(data_root))
        self.noisy_path = Path('{}/condition'.format(data_root))
        self.output_path = Path('{}/output'.format(data_root))

        self.inventory = generate_inventory(self.output_path, datatype)
        self.inventory = sorted(self.inventory)
        self.data_len = len(self.inventory)

    def __getitem__(self, index):

        if self.datatype == '.wav':
            clean, sr = torchaudio.load(self.clean_path/self.inventory[index])
            assert(sr==self.sample_rate)
            noisy, sr = torchaudio.load(self.noisy_path/self.inventory[index])
            assert (sr == self.sample_rate)
            output, sr = torchaudio.load(self.output_path/self.inventory[index])
            assert (sr == self.sample_rate)
        elif self.datatype == '.spec.npy' or self.datatype == '.mel.npy':
            raise NotImplementedError

        return clean, noisy, output




if __name__ == '__main__':

    try:
        import simpleaudio as sa
        hasAudio = True
    except ModuleNotFoundError:
        hasAudio = False
    sample_rate = 16000
    dataroot = '../data/Voicebank-DEMAND/train_28spk'
    datatype = '.wav'
    T = 7200
    dataset = AudioDataset(dataroot, datatype, sample_rate=sample_rate, T=T)
    dataloader = AudioDataLoader(dataset, batch_size=2, shuffle=True)
    clean, noisy, _ = next(iter(dataloader))
    print(clean.shape)  # should be [2, 1, T]

    play_obj = sa.play_buffer(clean[0, :, :].numpy(), 1, 32 // 8, sample_rate)
    play_obj.wait_done()

    play_obj = sa.play_buffer(noisy[0, :, :].numpy(), 1, 32 // 8, sample_rate)
    play_obj.wait_done()


