import os
import argparse
import soundfile as sf
import pandas as pd
import numpy as np
import functools

from pathlib import Path
from typing import List, Tuple, Union

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split
import random
from collections import defaultdict
import torchaudio.transforms as T
import librosa.effects

SampleType = Tuple[int, torch.Tensor, List[torch.Tensor]]

       
def compute_amplitude(waveforms, lengths=None, amp_type="avg", scale="linear"):
    if len(waveforms.shape) == 1:
        waveforms = waveforms.unsqueeze(0)

    assert amp_type in ["avg", "peak"]
    assert scale in ["linear", "dB"]

    if amp_type == "avg":
        if lengths is None:
            out = torch.mean(torch.abs(waveforms), dim=1, keepdim=True)
        else:
            wav_sum = torch.sum(input=torch.abs(waveforms), dim=1, keepdim=True)
            out = wav_sum / lengths
    elif amp_type == "peak":
        out = torch.max(torch.abs(waveforms), dim=1, keepdim=True)[0]
    else:
        raise NotImplementedError

    if scale == "linear":
        return out
    elif scale == "dB":
        return torch.clamp(20 * torch.log10(out), min=-80)  # clamp zeros
    else:
        raise NotImplementedError


def normalize(waveforms, lengths=None, amp_type="avg", eps=1e-14):

    assert amp_type in ["avg", "peak"]

    batch_added = False
    if len(waveforms.shape) == 1:
        batch_added = True
        waveforms = waveforms.unsqueeze(0)

    den = compute_amplitude(waveforms, lengths, amp_type) + eps
    if batch_added:
        waveforms = waveforms.squeeze(0)
    return waveforms / den


def rescale(waveforms, lengths, target_lvl, amp_type="avg", scale="linear"):

    assert amp_type in ["peak", "avg"]
    assert scale in ["linear", "dB"]

    batch_added = False
    if len(waveforms.shape) == 1:
        batch_added = True
        waveforms = waveforms.unsqueeze(0)

    waveforms = normalize(waveforms, lengths, amp_type)

    if scale == "linear":
        out = target_lvl * waveforms
    elif scale == "dB":
        out = dB_to_amplitude(target_lvl) * waveforms

    else:
        raise NotImplementedError("Invalid scale, choose between dB and linear")

    if batch_added:
        out = out.squeeze(0)

    return out 
def dB_to_amplitude(SNR):
    return 10 ** (SNR / 20)

class WSJ0Mix_DM(Dataset):

    def __init__(
        self,
        root: Union[str, Path],
        subset: str = "tr",
        num_speakers: int = 2,
        sample_rate: int = 8000,
        specaugment: bool = False,
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
        
        assert not specaugment 
        
        self.files = []
        for src_dir in src_dirs:
            self.files += [str(p) for p in src_dir.glob("**/*wav")]
        #print("Files:",self.files)
        
        _, sr_ori = torchaudio.load(self.files[0])
        self.resampler = T.Resample(sr_ori, sample_rate)
        
        self.files.sort()
        self.spk_dict = defaultdict(list)
        for path in self.files:
            self.spk_dict[self._get_speaker(path)].append(path)
        self.spk_list = list(self.spk_dict.keys())
        self.spk_weights = [len(self.spk_dict[spk]) / len(self.files) for spk in self.spk_list]
        
        # max amplitude in sources and mixtures
        self.MAX_AMP = 0.9
        # In LibriSpeech all the sources are at 16K Hz
        self.RATE = 16000
        # We will randomize loudness between this range
        self.MIN_LOUDNESS = -33
        self.MAX_LOUDNESS = -25
        self.MAX_FRAMES = 3200000
        
        self.specaugment = specaugment
            
    def _specaugment(self, sample):
        ffrate = random.uniform(0.95, 1.05)
        out = librosa.effects.time_stretch(sample, rate=ffrate)
        return out
        
    def _get_speaker(self, path:Union[str, Path]) -> str:
        speaker = os.path.split(os.path.dirname(str(path)))[1]
        return speaker

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, key: int) -> SampleType:
        """Load the n-th sample from the dataset.
        Args:
            key (int): The index of the sample to be loaded
        Returns:
            (int, Tensor, List[Tensor]): ``(sample_rate, mix_waveform, list_of_source_waveforms)``
        """
        return  self._dynamic_mixing()
    
    def _dynamic_mixing(self) -> SampleType:
        speakers = np.random.choice(self.spk_list, self.num_speakers, replace=False, p=self.spk_weights)

        sources, first_lvl = [], None

        spk_files = [np.random.choice(self.spk_dict[spk], 1, False)[0] for spk in speakers]
        
        sources = []
        for i, spk_file in enumerate(spk_files):
            tmp, _ = sf.read(spk_file, dtype='float32')
            if self.specaugment:
                tmp = self._specaugment(tmp)
            sources.append(torch.from_numpy(tmp).unsqueeze(0))

        minlen = min(*[source.size(dim=-1) for source in sources],self.MAX_FRAMES)

        sources_out = []
        for i, source in enumerate(sources):

            # select random offset
            length = source.size(dim=-1)
            start, stop = 0, length
            if length > minlen:  # take a random window
                start = np.random.randint(0, length - minlen)
                stop = start + minlen
            
            #print(source.shape, start, stop)
            tmp = source[:,start:stop]
            tmp = self.resampler(tmp)

            # peak = float(Path(spk_file).stem.split("_peak_")[-1])
            tmp = tmp[0]  # * peak  # remove channel dim and normalize

            if i == 0:
                gain = np.clip(random.normalvariate(-27.43, 2.57), -45, 0)
                tmp = rescale(tmp, torch.tensor(len(tmp)), gain, scale="dB")
                # assert not torch.all(torch.isnan(tmp))
                first_lvl = gain
            else:
                gain = np.clip(
                    first_lvl + random.normalvariate(-2.51, 2.66), -45, 0
                )
                tmp = rescale(tmp, torch.tensor(len(tmp)), gain, scale="dB")
                # assert not torch.all(torch.isnan(tmp))
            sources_out.append(tmp)

        sources = sources_out
        sources = torch.stack(sources)
        mixture = torch.sum(sources, 0)

        max_amp = max(
            torch.abs(mixture).max().item(),
            *[x.item() for x in torch.abs(sources).max(dim=-1)[0]],
        )
        mix_scaling = 1 / max_amp * 0.9
        sources = mix_scaling * sources
        mixture = mix_scaling * mixture

        return minlen, mixture.unsqueeze(0), [source.unsqueeze(0) for source in torch.unbind(sources, dim = 0)]

from tqdm import tqdm
if __name__ == "__main__":
    
    ds = WSJ0Mix_DM(
        root = "/data2/ohsai/WSJ0_wav/wsj0-wav/",
        subset = "cv",
        num_speakers = 3,
        sample_rate =  8000,
        specaugment = False,
    )
    generator = DataLoader(ds, batch_size=1, shuffle=False, num_workers=1)
    _, mix, [s1,s2, s3] = next(iter(generator))
    print(mix.shape)
    print(s1.shape)
    print(s2.shape)
    print(s3.shape)
    sr = 8000
    sf.write("mix.wav", mix[0,0,:].numpy(), sr)
    sf.write("s1.wav", s1[0,0,:].numpy(), sr)
    sf.write("s2.wav", s2[0,0,:].numpy(), sr)
    sf.write("s3.wav", s3[0,0,:].numpy(), sr)
    
