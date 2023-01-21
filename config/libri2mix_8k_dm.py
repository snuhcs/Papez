from core.moduledict import ModuleDict, Munch
from core.dataset import LibriMix_DM, LibriMix
rootdir = "/data1/ohsai/LibriMix/"
num_speakers = 2
sample_rate = 8000
config = Munch(
    train_dataset = ModuleDict(
        module = LibriMix_DM,
        root = "/data1/ohsai/LibriMix/",
        root_librispeech = "/data1/ohsai/LibriSpeech/LibriSpeech",
        root_wham = None,
        metadata_dir = "/data1/ohsai/LibriSpeech/metadata",
        subset = "mix_clean",
        num_speakers = 2,
        sample_rate = 8000,
    ),
    valid_dataset = ModuleDict(
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