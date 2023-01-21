import os
from core.moduledict import ModuleDict, Munch, import_config_from_path
from core.model import Sepformer_Original
import torch.nn as nn

num_speakers = 2
sample_rate = 8000
tag_suffix = "librimix_original"

config = import_config_from_path("config/libri2mix_8k.py", "config/optim_adam.py")
config.update(Munch(
    sample_rate = sample_rate,
    num_speakers = num_speakers,
    save_directory = os.path.join("work_dir/",
                                  f'papez/{num_speakers}ppl/{sample_rate}HZ/',
                                  tag_suffix),
    checkpoint_step = 4000,
    evaluation_step = 100,
    total_steps = 10_000_000,
    model = ModuleDict(
        module = Sepformer_Original,
        intra_model = lambda : nn.TransformerEncoder(
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=256, 
                nhead=8, 
                dim_feedforward=1024, 
                dropout=0.1,
                batch_first=True, 
                norm_first=False,
            ),
            num_layers = 4,
        ),
        inter_model = lambda : nn.TransformerEncoder(
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=256, 
                nhead=8, 
                dim_feedforward=1024, 
                dropout=0.1,
                batch_first=True, 
                norm_first=False,
            ),
            num_layers = 4,
        ),
        masknet_numspks=num_speakers,
    ),
))