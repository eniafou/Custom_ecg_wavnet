"""
Show raw audio and mu-law encode samples to make input source.
"""
import os

import numpy as np
import wfdb
import torch
import torch.utils.data as data
import pandas as pd


def load_audio(filename):
    # sample rate is not important, it was used in the audio implementation of this function.
    signal, meta = wfdb.rdsamp(filename)

    return signal


def one_hot_encode(data, channels=256):
    one_hot = np.zeros((data.size, channels), dtype=float)
    one_hot[np.arange(data.size), data.ravel()] = 1

    return one_hot


def one_hot_decode(data, axis=1):
    decoded = np.argmax(data, axis=axis)

    return decoded


def mu_law_encode(audio, quantization_channels=256):
    """
    Quantize waveform amplitudes.
    Reference: https://github.com/vincentherrmann/pytorch-wavenet/blob/master/audio_data.py
    """
    mu = float(quantization_channels - 1)
    quantize_space = np.linspace(-1, 1, quantization_channels)

    quantized = np.sign(audio) * np.log(1 + mu * np.abs(audio)) / np.log(mu + 1)
    quantized = np.digitize(quantized, quantize_space) - 1

    return quantized


def mu_law_decode(output, quantization_channels=256):
    """
    Recovers waveform from quantized values.
    Reference: https://github.com/vincentherrmann/pytorch-wavenet/blob/master/audio_data.py
    """
    mu = float(quantization_channels - 1)

    expanded = (output / quantization_channels) * 2. - 1
    waveform = np.sign(expanded) * (
                   np.exp(np.abs(expanded) * np.log(mu + 1)) - 1
               ) / mu

    return waveform


class RawDataset(data.Dataset):
    def __init__(self, data_dir, receptive_fields = 0, start = 0, sample_size = 100, data_len = 100, istraining = True):
        super(RawDataset, self).__init__()
        self.istraining = istraining
        self.receptive_fields = receptive_fields
        self.start = start
        self.sample_size = sample_size
        self.root_path = data_dir
        self.filenames = pd.read_csv(data_dir+'ptbxl_database.csv', index_col='ecg_id')["filename_lr"].iloc[:data_len]
    
    @staticmethod
    def _variable(data, transpose = False):
        tensor = torch.from_numpy(data).float()
        if transpose:
            tensor = tensor.transpose(0,1)
        if torch.cuda.is_available():
            return torch.autograd.Variable(tensor.cuda())
        else:
            return torch.autograd.Variable(tensor)
        

    def __getitem__(self, index):
        filepath = os.path.join(self.root_path, self.filenames.iloc[index])

        raw_audio = load_audio(filepath)
        raw_audio = raw_audio[self.start:self.sample_size, :]
        if self.istraining:

            h = raw_audio[:,[0]] # shape (1000,1)
            h = np.pad(h, [[self.receptive_fields, 0], [0,0]], 'constant')
        
            x = raw_audio[:,[1]] # shape (1000,1)
            
            target = x.copy()
            
            
            x = x[:-1]
            x = np.pad(x, [[self.receptive_fields + 1, 0], [0,0]], 'constant')
            
            return (self._variable(h, True),self._variable(x, True)), self._variable(target)
        
        return raw_audio

    def __len__(self):
        return len(self.filenames)
    

class Dataset(data.Dataset):
    def __init__(self, data_dir, receptive_fields, in_channels=256, data_len = 100):
        super(Dataset, self).__init__()

        self.in_channels = in_channels
        self.receptive_fields = receptive_fields
        self.root_path = data_dir
        self.filenames = pd.read_csv(data_dir+'ptbxl_database.csv', index_col='ecg_id')["filename_lr"].iloc[:data_len]

    @staticmethod
    def _variable(data, transpose = False):
        tensor = torch.from_numpy(data).float()
        if transpose:
            tensor = tensor.transpose(0,1)
        if torch.cuda.is_available():
            return torch.autograd.Variable(tensor.cuda())
        else:
            return torch.autograd.Variable(tensor)
        
    def __getitem__(self, index):
        filepath = os.path.join(self.root_path, self.filenames.iloc[index])

        raw_audio = load_audio(filepath)

        h = raw_audio[:,0] # shape (1000,)
        h = np.pad(h, [[self.receptive_fields, 0]], 'constant')
        h = mu_law_encode(h, self.in_channels)
        h = one_hot_encode(h, self.in_channels)
       
        x = raw_audio[:,1] # shape (1000,)
        
        target = mu_law_encode(x, self.in_channels)
        target = one_hot_encode(target, self.in_channels)
        
        x = x[:-1]
        x = np.pad(x, [[self.receptive_fields + 1, 0]], 'constant')
        x = mu_law_encode(x, self.in_channels)
        x = one_hot_encode(x, self.in_channels)
        return (self._variable(h, True),self._variable(x, True)), self._variable(one_hot_decode(target, 1))
    

    def __len__(self):
        return len(self.filenames)

if __name__ == '__main__':
    dataset = Dataset(data_dir = "../data/ptb-xl/")
    print(dataset[0])