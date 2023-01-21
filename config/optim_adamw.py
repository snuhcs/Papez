from core.moduledict import ModuleDict, Munch
import torch.optim
import torch.optim.lr_scheduler
config = Munch(
    optimizer = ModuleDict(
        module = torch.optim.AdamW,
        submodules = [
            ModuleDict(
                module = torch.optim.lr_scheduler.ExponentialLR,
                gamma=0.99, 
            )
        ],
        lr = 1e-4,
        weight_decay = 1e-4
    ),
)