import os
import argparse
import soundfile as sf
import pandas as pd
import numpy as np
import functools
from scipy.signal import resample_poly
import tqdm.contrib.concurrent

from pathlib import Path
from typing import List, Tuple, Union

import torch
import torchaudio
from torch.utils.data import Dataset
import pyloudnorm as pyln
import random
import torchaudio.transforms as T
import librosa.effects
from torch.utils.data import DataLoader, random_split

SampleType = Tuple[int, torch.Tensor, List[torch.Tensor]]


class LibriMix_DM(Dataset):

    def __init__(
        self,
        root: Union[str, Path],
        root_librispeech: Union[str, Path],
        root_wham: Union[str, Path],
        metadata_dir: Union[str, Path],
        subset: str = "mix_clean",
        num_speakers: int = 2,
        sample_rate: int = 8000,
        specaugment: bool = False,
    ):
        assert subset in ['mix_clean', 'mix_both','mix_single']
        assert not(subset != 'mix_clean' and root_wham in [None, ""])
        self.mode = 'min'
        self.sample_rate = sample_rate
        
        self.task = subset
        self.num_speakers = num_speakers
        self.librispeech_files, self.wham_files = self._load_sources(metadata_dir, root_librispeech, root_wham, subset)
        self.librispeech_files = sorted(self.librispeech_files, key=lambda x: x['speaker_ID'])
        
        # max amplitude in sources and mixtures
        self.MAX_AMP = 0.9
        # In LibriSpeech all the sources are at 16K Hz
        self.RATE = 16000
        # We will randomize loudness between this range
        self.MIN_LOUDNESS = -33
        self.MAX_LOUDNESS = -25
        
        assert not specaugment 
        self.specaugment = specaugment
            
    def _specaugment(self, sample):
        ffrate = random.uniform(0.95, 1.05)
        out = librosa.effects.time_stretch(sample, rate=ffrate)
        return out

    def __len__(self) -> int:
        return len(self.librispeech_files)

    def __getitem__(self, key: int) -> SampleType:
        """Load the n-th sample from the dataset.
        Args:
            key (int): The index of the sample to be loaded
        Returns:
            (int, Tensor, List[Tensor]): ``(sample_rate, mix_waveform, list_of_source_waveforms)``
        """
        return self._dynamic_mixing(key, self.num_speakers, self.task)
    
    def _load_sources(self, metadata_dir: Union[str, Path], root_librispeech: Union[str, Path], root_wham: Union[str, Path], task: str):
        librispeech_md_dir = Path(metadata_dir) / "LibriSpeech"
        librispeech_md_files = os.listdir(str(librispeech_md_dir))
        #print(librispeech_md_dir)
        librispeech_md_file = [f for f in librispeech_md_files if 'train-clean' in f][0] # train-clean-360: 0, train-clean-100: 1
        
        if not librispeech_md_file.endswith('.csv'):
            raise ValueError(f"{librispeech_md_file} is not a csv file, continue.")
            
        librispeech_md = pd.read_csv(str(librispeech_md_dir/ librispeech_md_file), engine='python')
        librispeech_md['origin_path'] = librispeech_md['origin_path'].apply(lambda x: str(Path(root_librispeech) /x) )
        librispeech_files = librispeech_md.to_dict('records') 
        
        wham_files = None
        if task != "mix_clean":
            wham_md_dir = Path(metadata_dir) / "Wham_noise"
            wham_md_files = os.listdir(str(wham_md_dir))
            try:
                wham_md_file = [f for f in wham_md_files if
                                f.startswith(librispeech_md_file.split('-')[0])][0]
            except IndexError:
                raise ValueError('Wham metadata are missing you can either generate the '
                      'missing wham files or add the librispeech metadata to '
                      'to_be_ignored list')
            wham_md = pd.read_csv(str(wham_md_dir, wham_md_file), engine='python')
            wham_md['origin_path'] = wham_md['origin_path'].apply(lambda x: str(Path(root_wham) /x) )
            wham_files = wham_md.to_dict('records') 
            
        return librispeech_files, wham_files
    
    def _dynamic_mixing(self, key:int, n_src:int, task:str) -> SampleType:
        keys = [key]
        while len(keys) < n_src:
            keys.append(self._sample_key(self.librispeech_files, keys))
        
        #print("Key 1, Key 2, len", key, key2, len(self.librispeech_files))
        # Get sources and mixture infos
        sources_info, sources = self.load_sources(self.librispeech_files, keys, n_src)
        
        if task == 'mix_clean':
            sources_to_mix = sources[:n_src]
            use_noise = False
        else:
            key_random = random.randint(0,len(self.wham_files) -1)
            noises_info, noises = self.load_sources(self.wham_files, [key_random], 1)
            sources_info += noises_info
            use_noise = True
            if task == 'mix_both':
                sources_to_mix = sources + noises
            elif task == 'mix_single':
                sources_to_mix = [sources[0], noises[0]]
            else:
                raise ValueError("Task should be in ['mix_clean', 'mix_both', 'mix_single']")
        
        sources_norm  = self.randomize_loudness(sources_to_mix, use_noise = use_noise)
        sources_resampled = self.resample(sources_norm, self.sample_rate)
        sources_reshaped = self.match_length(sources_resampled, self.mode)
        
        mixture = self.mix(sources_reshaped)
        #renormalize_loudness, did_clip = self.check_for_cliping(mixture, sources_reshaped)
        sources_reshaped = [torch.from_numpy(source).unsqueeze(0).float() for source in sources_reshaped]
        #print(sources_reshaped[0].shape)

        return sources_info, mixture.unsqueeze(0).float(), sources_reshaped
    def _sample_key(self, files, keys:int) -> int:
        #print(files[keys[0]].keys())
        spks = [files[key]['speaker_ID'] for key in keys]
        key_new = random.randint(0,len(files) -1)
        spk_new = files[key_new]['speaker_ID']
        while spk_new in spks or key_new in keys:
            key_new = random.randint(0,len(files) -1)
            spk_new = files[key_new]['speaker_ID']
        return key_new
        
    
    def load_sources(self, files, keys, n_src):
        # Read lines corresponding to pair
        sources = [files[keys[i]] for i in range(n_src)]
                
        sources_list = []
        # Read the source and compute some info
        for i in range(n_src):
            absolute_path = sources[i]['origin_path']
            s, _ = sf.read(absolute_path, dtype='float32')
            if self.specaugment:
                s = self._specaugment(s)
            sources_list.append(s)

        sources_info = sources
        return sources_info, sources_list

    def resample(self, sources_list, freq): # Rate of the sources in LibriSpeech
        """ Resample the source list to the desired frequency"""
        # Create the resampled list
        resampled_list = []
        # Resample each source
        for source in sources_list:
            resampled_list.append(resample_poly(source, freq, self.RATE))
        return resampled_list


    def match_length(self,source_list, mode):
        """ Make the sources to match the target length """
        sources_list_reshaped = []
        # Check the mode
        if mode == 'min':
            target_length = min([len(source) for source in source_list])
            for source in source_list:
                start = np.random.randint(0, max(len(source) -  target_length, 0) + 1)
                sources_list_reshaped.append(source[start:start + target_length])
        else:
            target_length = max([len(source) for source in source_list])
            for source in source_list:
                sources_list_reshaped.append(
                    np.pad(source, (0, target_length - len(source)),
                           mode='constant'))
        return sources_list_reshaped

    def randomize_loudness(self,sources, use_noise:bool):
        """ Compute original loudness and normalise them randomly """
        # In LibriSpeech all sources are at 16KHz hence the meter
        meter = pyln.Meter(self.RATE)
        # Randomize sources loudness
        sources_norm = []

        # Normalize loudness
        for i, source in enumerate(sources):
            # Compute initial loudness
            loudness = meter.integrated_loudness(source)
            # Pick a random loudness
            target_loudness = random.uniform(self.MIN_LOUDNESS, self.MAX_LOUDNESS)
            # Noise has a different loudness
            if use_noise and (i == len(source) -1):
                target_loudness = random.uniform(self.MIN_LOUDNESS - 5,
                                                 self.MAX_LOUDNESS - 5)
            # Normalize source to target loudness

            src = pyln.normalize.loudness(source, loudness, target_loudness)
            # If source clips, renormalize
            
            if np.max(np.abs(src)) >= 1:
                src = source * self.MAX_AMP / np.max(np.abs(source))
                target_loudness = meter.integrated_loudness(src)
            
            # Save scaled source and loudness.
            sources_norm.append(src)
        return sources_norm 


    def mix(self, sources):
        sources = [torch.from_numpy(source) for source in sources]
        mixture = torch.zeros_like(sources[0])
        for i, source in enumerate(sources):
            mixture += source
        return mixture

if __name__ == "__main__":
    ds = LibriMix_DM(
        root = "/data1/ohsai/LibriMix/",
        root_librispeech = "/data2/ohsai/LibriSpeech/LibriSpeech",
        root_wham = None,
        metadata_dir = "/data2/ohsai/LibriSpeech/metadata",
        subset = "mix_clean",
        num_speakers = 3,
        sample_rate = 8000,
        specaugment = False,
    )
    generator = DataLoader(ds, batch_size=1, shuffle=True, num_workers=1)
    e, mix, [s1,s2,s3] = next(iter(generator))
    print(mix.shape, s1.shape, s2.shape,s3.shape)
    sr = 8000
    sf.write("mix.wav", mix[0,0,:].numpy(), sr)
    sf.write("s1.wav", s1[0,0,:].numpy(), sr)
    sf.write("s2.wav", s2[0,0,:].numpy(), sr)
    sf.write("s3.wav", s3[0,0,:].numpy(), sr)
    
