import os
from core.moduledict import ModuleDict2 as ModuleDict
from core.moduledict import Munch, import_config_from_path
from core.model import AB_Papez_OuterStep, AB_Trans_Recurrent
import torch.nn as nn

num_speakers = 2
sample_rate = 8000
tag_suffix = "wsj0mix/ablation_recurrent"

config = import_config_from_path("config/wsj02mix_8k_dc.py", "config/optim_adam.py")
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
        module = AB_Papez_OuterStep,
        submodules = Munch(
            intra_model = ModuleDict(
                module = AB_Trans_Recurrent,
                submodules = Munch(
                    encoder_layer = ModuleDict(
                        module = nn.TransformerEncoderLayer,
                        d_model=256, 
                        nhead=8, 
                        dim_feedforward=1024, 
                        dropout=0.1,
                        batch_first=True, 
                        norm_first=False,
                    ),
                ),
                num_layers = 4,
                hidden_size = 256,
                max_seq_len=250,
            ),
            inter_model = ModuleDict(
                module = AB_Trans_Recurrent,
                submodules = Munch(
                    encoder_layer = ModuleDict(
                        module = nn.TransformerEncoderLayer,
                        d_model=256, 
                        nhead=8, 
                        dim_feedforward=1024, 
                        dropout=0.1,
                        batch_first=True, 
                        norm_first=False,
                    ),
                ),
                num_layers = 4,
                hidden_size = 256,
                max_seq_len=300,
            ),
        ),
        masknet_numspks=num_speakers,
        masknet_chunksize=250,
    ),
))