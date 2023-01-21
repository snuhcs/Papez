from core.moduledict import ModuleDict, Munch
from core.dataset import LibriMix
rootdir = "/data1/ohsai/LibriMix/"
num_speakers = 2
sample_rate = 8000
config = Munch(
    train_dataset = ModuleDict(
        module = LibriMix,
        root = rootdir,
        subset = "train-360",
        num_speakers = num_speakers,
        sample_rate = sample_rate,
        task = 'sep_clean',
    ),
    test_dataset = ModuleDict(
        module = LibriMix,
        root = rootdir,
        subset = "test",
        num_speakers = num_speakers,
        sample_rate = sample_rate,
        task = 'sep_clean',
    ),
)