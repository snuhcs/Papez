from core.moduledict import ModuleDict, Munch
import torch.optim
import torch.optim.lr_scheduler
config = Munch(
    optimizer = ModuleDict(
        module = torch.optim.Adam,
        submodules = [
            ModuleDict(
                module = torch.optim.lr_scheduler.StepLR,
                step_size = 100,
                gamma = 0.33333, 
            )
        ],
        lr = 1e-3,
    ),
)