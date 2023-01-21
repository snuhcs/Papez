from config.papez_study_libri2mix import config

import torch
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl

import argparse
import main, os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-gpu", "--gpu", type=str, default='0') 
    parser.add_argument("-tags", "--tags", type=str, default='dummy') 
    args = parser.parse_args()
    if isinstance(args.gpu, str):
        args.gpu = [int(n) for n in args.gpu.split(",")]
        
    # Initialize loggers
    tags = args.tags.split(",")
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="tb_logs/")
    
    boilerplate = main.LitModule(config, seed = 42)

    if "ckpt_path" in config:
        if os.path.exists(config.ckpt_path):
            boilerplate = boilerplate.load_from_checkpoint(config.ckpt_path)
        else:
            print("** Given checkpoint path does not exist!")
    else:
        print("** No checkpoint path given!")
    
    # Model Sanity Check
    #sample = boilerplate.model(next(iter(test_loader))[1])

    # Test the model
    trainer = pl.Trainer(precision=16,limit_test_batches=7,
                        default_root_dir=config["save_directory"],
                        accelerator="gpu", gpus = args.gpu,
                        logger=[tb_logger])

    trainer.test(model= boilerplate, #bp2,
                )