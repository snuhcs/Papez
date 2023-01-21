from core.moduledict import ModuleDict, Munch
from core.dataset import WSJ0Mix, WSJ0Mix_DM
from torch.utils.data import DataLoader, random_split
rootdir = "/data1/ohsai/WSJ0/WSJ0_2Mix_DC/"
num_speakers = 2
sample_rate = 8000
config = Munch(
    train_dataset = ModuleDict(
        module = WSJ0Mix_DM,
        root = "/data1/ohsai/WSJ0/wsj0-wav/",
        subset = "tr",
        num_speakers = num_speakers,
        sample_rate = sample_rate,
    ),
    valid_dataset = ModuleDict(
        module = WSJ0Mix,
        root = rootdir,
        subset = "cv",
        num_speakers = num_speakers,
        sample_rate = sample_rate,
    ),
    test_dataset = ModuleDict(
        module = WSJ0Mix,
        root = rootdir,
        subset = "tt",
        num_speakers = num_speakers,
        sample_rate = sample_rate,
    ),
)