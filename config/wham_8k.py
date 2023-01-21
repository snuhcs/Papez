from core.moduledict import ModuleDict, Munch
from core.dataset import WHAM
rootdir = '/data2/ohsai/WHAM/wham_dataset/'
num_speakers = 2
sample_rate = 8000
timelength = 4.0
config = Munch(
    train_dataset = ModuleDict(
        module = WHAM,
        root_dirpath=rootdir, 
        split='tr', 
        task='sep_clean',
        sample_rate=sample_rate, 
        timelength=timelength,
        zero_pad=True, 
        min_or_max='min', 
        augment=True,
        normalize_audio=True, 
        n_samples=0
    ),
    test_dataset = ModuleDict(
        module = WHAM,
        root_dirpath=rootdir, 
        split='tt', 
        task='sep_clean',
        sample_rate=sample_rate, 
        timelength=timelength,
        zero_pad=False, 
        min_or_max='min', 
        augment=False,
        normalize_audio=True, 
        n_samples=0
    ),
)
