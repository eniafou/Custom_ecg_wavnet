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
    signal, meta = wfdb.rdsamp(filename)
    return signal


def one_hot_encode(data, channels=256):
    one_hot = np.zeros((data.size, channels), dtype=float)
    one_hot[np.arange(data.size), data.ravel()] = 1

    one_hot = np.transpose(one_hot, (1, 0))
    return one_hot


def one_hot_decode(data, axis=0):
    decoded = np.argmax(data, axis=axis)
    return decoded


def mu_law_encode(audio, quantization_channels=256):
    mu = float(quantization_channels - 1)
    quantized = np.sign(audio) * np.log(1 + mu * np.abs(audio)) / np.log(mu + 1)
    
    quantized = quantize(quantized, quantization_channels)
    return quantized

def quantize(signal, quantization_channels = 256):
    def fun(x):
        y = x - 1
        if y>=0:
            return y
        return 0
    
    vectorized_func = np.vectorize(fun)
    quantize_space = np.linspace(-1, 1, quantization_channels)

    quantized = vectorized_func(np.digitize(signal, quantize_space))

    return quantized

def expand(signal, quantization_channels = 256):
    return (signal / quantization_channels) * 2. - 1


def mu_law_decode(output, quantization_channels=256):
    mu = float(quantization_channels - 1)

    expanded = expand(output, quantization_channels)
    waveform = np.sign(expanded) * (
                   np.exp(np.abs(expanded) * np.log(mu + 1)) - 1
               ) / mu

    return waveform


class RawDataset(data.Dataset):
    def __init__(self, data_dir, receptive_field = 0, start = 0, sample_size = 100, data_len = 100, istraining = True, conditioned = True, freq = "hr"):
        super(RawDataset, self).__init__()

        self.freq = freq
        self.conditioned = conditioned
        self.istraining = istraining
        self.receptive_field = receptive_field
        self.start = start
        self.sample_size = sample_size
        self.root_path = data_dir
        self.filenames = pd.read_csv(data_dir+'ptbxl_database.csv', index_col='ecg_id')["filename_" + str(freq)].iloc[:data_len]
    
    @staticmethod
    def _variable(data):
        tensor = torch.from_numpy(data).float()
        if torch.cuda.is_available():
            return torch.autograd.Variable(tensor.cuda())
        else:
            return torch.autograd.Variable(tensor)
        

    def __getitem__(self, index):
        filepath = os.path.join(self.root_path, self.filenames.iloc[index])

        raw_audio = load_audio(filepath)
        raw_audio = raw_audio[self.start:self.sample_size, :]
        if self.istraining:

            h = raw_audio[:,0] # shape (1000,)
            h = np.pad(h, [[self.receptive_field, 0]], 'constant')
        
            target = raw_audio[:,1] # shape (1000,)
            
            
            if self.conditioned:
                x = target.copy()
                x = x[:-1]
                x = np.pad(x, [[self.receptive_field + 1, 0]], 'constant')
                
                return (self._variable(h),self._variable(x)), self._variable(target)
            
            return self._variable(h),self._variable(target)


        
        return raw_audio

    def __len__(self):
        return len(self.filenames)
    

class Dataset(data.Dataset):
    def __init__(self, data_dir, receptive_field = 0, in_channels=256,start = 0, sample_size = 1000, data_len = 100, conditioned = True, freq = "hr"):
        super(Dataset, self).__init__()
        self.freq = freq
        self.start = start
        self.sample_size = sample_size
        self.conditioned = conditioned
        self.in_channels = in_channels
        self.receptive_field = receptive_field
        self.root_path = data_dir
        self.filenames = pd.read_csv(data_dir+'ptbxl_database.csv', index_col='ecg_id')["filename_" + str(freq)].iloc[:data_len]

    @staticmethod
    def _variable(data):
        tensor = torch.from_numpy(data).float()

        if torch.cuda.is_available():
            return torch.autograd.Variable(tensor.cuda())
        else:
            return torch.autograd.Variable(tensor)
        
    def __getitem__(self, index):
        filepath = os.path.join(self.root_path, self.filenames.iloc[index])

        raw_audio = load_audio(filepath)
        raw_audio = raw_audio[self.start:self.sample_size, :]

        h = raw_audio[:,0] # shape (1000,)
        h = np.pad(h, [[self.receptive_field, 0]], 'constant')
        h = mu_law_encode(h, self.in_channels)
        h = one_hot_encode(h, self.in_channels)
       
        x = raw_audio[:,1] # shape (1000,)
        
        target = mu_law_encode(x, self.in_channels)
        
        if self.conditioned:
            x = x[:-1]
            x = np.pad(x, [[self.receptive_field + 1, 0]], 'constant')
            x = mu_law_encode(x, self.in_channels)
            x = one_hot_encode(x, self.in_channels)
            return (self._variable(h),self._variable(x)), self._variable(target)
        
        return self._variable(h),self._variable(target)

    def __len__(self):
        return len(self.filenames)




if __name__ == '__main__':
    # fix the dataset, it should be optimal of pytorch use
    # input (C,d, ..)   target (d,d,....)
    dataset = Dataset("../data/ptb-xl/", 10, in_channels=256, data_len = 100, conditioned=False)
    y = dataset[3][1]
    # print(e[0][0].shape, e[0][1].shape, e[1].shape)
    # y_trad = mu_law_encode(y, 256)
    for i in y:
        if i<0:
            print(i)
    



    



    
    