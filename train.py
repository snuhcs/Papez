from config.papez_study_libri2mix import config 
import pytorch_lightning as pl

import torch
from pytorch_lightning import loggers as pl_loggers

import argparse
import main
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-tags", "--tags", type=str, default='dummy') 
    parser.add_argument("-gpu", "--gpu", type=str, default='0') 
    
    args = parser.parse_args()
    tags = args.tags.split(",")
    if isinstance(args.gpu, str):
        args.gpu = [int(n) for n in args.gpu.split(",")]
    
    # Initialize loggers
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="tb_logs/")
    
    # Setup Pytorch lightning module
    boilerplate = main.LitModule(config)

    if "ckpt_path" in config:
        if os.path.exists(config.ckpt_path):
            ckpt_path = config.ckpt_path
        else:
            print("** Given checkpoint path does not exist!")
            ckpt_path = None
    else:
        print("** No checkpoint path given!")
        ckpt_path = None

    # train the model
    trainer = pl.Trainer(precision=16, 
                         limit_val_batches=200, 
                        max_epochs = 1000,
                        default_root_dir=config["save_directory"],
                        accelerator="gpu", devices = args.gpu,
                        logger=[tb_logger],
                        gradient_clip_val=1.0,
                        num_sanity_val_steps=2,
                        resume_from_checkpoint=ckpt_path,
                        )
    
    trainer.fit(model=boilerplate)
