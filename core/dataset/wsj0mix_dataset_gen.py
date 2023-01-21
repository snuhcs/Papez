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
from torch.utils.data import DataLoader, random_split
import pyloudnorm as pyln
import random

SampleType = Tuple[int, torch.Tensor, List[torch.Tensor]]


class WSJ0Mix_Gen(Dataset):

    def __init__(
        self,
        root: Union[str, Path],
        subset: str = "mix_clean",
        num_speakers: int = 2,
        sample_rate: int = 8000,
    ):
        assert subset in ['tr', 'cv','tt']
        root = Path(root) / "wsj0"
        self.mode = 'min'
        self.sample_rate = sample_rate
        
        self.task = subset
        self.num_speakers = num_speakers
        
        if subset == 'tr':
            subdirs = ["si_tr_s",]
        elif subset in ['cv', 'tt']:
            subdirs = ["si_dt_05", "si_et_05"]
        src_dirs = [root / subdir for subdir in subdirs]
        print("SRCDIRS:",src_dirs)
        assert os.path.exists(src_dirs[0])
        
        self.gender = {}
        with open("wsj0_gender_list.txt", 'r') as f_gender:
            lines = f_gender.readlines()
            for line in lines:
                speaker, gender = line.split()
                self.gender[speaker] = gender
        
        
        self.files = []
        for src_dir in src_dirs:
            self.files += [str(p) for p in src_dir.glob("**/*wav")]
        print("Files:",self.files)
        self.files.sort()
        
        # max amplitude in sources and mixtures
        self.MAX_AMP = 0.9
        # In LibriSpeech all the sources are at 16K Hz
        self.RATE = 16000
        # We will randomize loudness between this range
        self.MIN_LOUDNESS = -33
        self.MAX_LOUDNESS = -25

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, key: int) -> SampleType:
        """Load the n-th sample from the dataset.
        Args:
            key (int): The index of the sample to be loaded
        Returns:
            (int, Tensor, List[Tensor]): ``(sample_rate, mix_waveform, list_of_source_waveforms)``
        """
        return self._dynamic_mixing(key, self.num_speakers, self.task)
    
    def _dynamic_mixing(self, key:int, n_src:int, task:str) -> SampleType:
        key2 = self._sample_key_without_gender_overlap(key)
        #print("Key 1, Key 2, len", key, key2, len(self.files))
        # Get sources and mixture infos
        sources_info, sources = self.load_sources(self.files, [key,key2], n_src)
        
        sources_to_mix = sources[:n_src]
        use_noise = False
        
        sources_norm  = self.randomize_loudness(sources_to_mix, use_noise = use_noise)
        sources_resampled = self.resample(sources_norm, self.sample_rate)
        sources_reshaped = self.match_length(sources_resampled, self.mode)
        
        mixture = self.mix(sources_reshaped)
        renormalize_loudness, did_clip = self.check_for_cliping(mixture, sources_reshaped)
        sources_reshaped = [source for source in sources_reshaped]
        #print(sources_reshaped[0].shape)

        return sources_info, mixture, sources_reshaped
    def _sample_key_without_gender_overlap(self, key:int) -> int:
        speaker1 = os.path.split(os.path.dirname(self.files[key]))[1]
        gender1 = self.gender[speaker1]
        
        key2 = random.randint(0,len(self.files) -1)
        speaker2 = os.path.split(os.path.dirname(self.files[key2]))[1]
        gender2 = self.gender[speaker2]
        #print("gender1:", gender1, "gender2:",gender2)
        assert gender1 in ["m", "f"] and gender2 in ["m","f"]
        while key2 == key or gender1 == gender2:
            key2 = random.randint(0,len(self.files) -1)
            speaker2 = os.path.split(os.path.dirname(self.files[key2]))[1]
            gender2 = self.gender[speaker2]
            #print("gender2:",gender2)
            assert gender2 in ["m","f"]
        return key2
    
    def load_sources(self, files, keys, n_src):
        # Read lines corresponding to pair
        sources = [files[keys[i]] for i in range(n_src)]
                
        sources_list = []
        # Read the source and compute some info
        for i in range(n_src):
            absolute_path = sources[i]
            s, _ = sf.read(absolute_path)
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
                sources_list_reshaped.append(source[:target_length])
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
        mixture = np.zeros_like(sources[0])
        for source in sources:
            mixture += source
        return mixture


    def check_for_cliping(self, mixture, sources):
        """Check the mixture (mode max) for clipping and re normalize if needed."""
        renormalize_loudness = []
        clip = False
        meter = pyln.Meter(self.RATE)
        
        if np.max(np.abs(mixture)) > self.MAX_AMP: # Check for clipping in mixtures
            clip = True
            weight = self.MAX_AMP / np.max(np.abs(mixture))
        else:
            weight = 1
       
        for source in sources:
            new_loudness = meter.integrated_loudness(source * weight)  # Renormalize
            renormalize_loudness.append(new_loudness)
        return renormalize_loudness, clip

from tqdm import tqdm
if __name__ == "__main__":
    subset = "cv"
    output_dir = "/data1/ohsai/WSJ0/WSJ0_2Mix_mine"
    sr = 8000
    num_speakers = 2
    num_dataset = {"tr":50000, "cv":10000, "tt":5000}
    assert subset in num_dataset.keys()
    
    ds = WSJ0Mix_Gen(
        root = "/data2/ohsai/WSJ0_wav/wsj0-wav/",
        subset = subset,
        num_speakers = num_speakers,
        sample_rate = sr,
    )
    generator = DataLoader(ds, batch_size=1, shuffle=False, num_workers=1)
    
    output_dir = Path(output_dir)
    os.makedirs(str(output_dir / subset / "mix" ), exist_ok=True)
    os.makedirs(str(output_dir / subset / "s1" ), exist_ok=True)
    os.makedirs(str(output_dir / subset / "s2" ), exist_ok=True)
    i = 0
    while i < num_dataset[subset]:
        for e, mix, sources in tqdm(generator):
            s1, s2 = sources
            sf.write(output_dir / subset / "mix" / f'{i}.wav', mix[0].numpy(), sr)
            sf.write(output_dir / subset / "s1" / f'{i}.wav', s1[0].numpy(), sr)
            sf.write(output_dir / subset / "s2" / f'{i}.wav', s2[0].numpy(), sr)
            i = i + 1
            if i > num_dataset[subset]:
                break
    
