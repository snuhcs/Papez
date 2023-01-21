import pytorch_lightning as pl

import torch
from torch import optim, nn
from torchmetrics import PermutationInvariantTraining
import torchmetrics.functional
from torchmetrics.audio import pesq
from torch.utils.data import DataLoader, random_split

from config.papez_study_libri2mix import config as config2

import core.visualize as vis
import random


class LitModule(pl.LightningModule): # define the LightningModule
    def __init__(self, config, seed = None):
        super().__init__()
        self.save_hyperparameters(config, seed)
        
        if isinstance(seed, int):
            self.seed = pl.utilities.seed.seed_everything(seed)
        else:
            self.seed = pl.utilities.seed.seed_everything()
        print("LitModule seed:", self.seed)
            
        self.config = config
        
        self.model = config.model() # convtasnet
        
        for v in self.model.named_modules():
            print(v)
        
        

    def training_step(self, batch, batch_idx):
        #print(batch)
        _, x, y = batch
        y = torch.cat(y,dim = 1)
        
        pred = self.model(x)
        
        si_sdr = self.si_sdr_pit(y, pred)
        si_snr = self.si_snr_pit(y, pred)
        loss = -si_snr
        
        self.log("loss", loss)
        self.log("sisdr", si_sdr)
        self.log("sisnr", si_snr)
        return loss
    
    def validation_step(self, batch, batch_idx):
        _, x, y = batch
        y = torch.cat(y,dim = 1)
        
        pred = self.model(x)
        print("pred:", pred.shape, "y:", y.shape)
        
        si_sdr = self.si_sdr_pit(y, pred)
        si_snr = self.si_snr_pit(y, pred)
        loss = -si_snr
        
        self.log("valid_loss", loss, on_step=True, on_epoch=True)
        self.log("valid_sisdr", si_sdr, on_step=True, on_epoch=True)
        self.log("valid_sisnr", si_snr, on_step=True, on_epoch=True)
        
        try:
            sdr = self.sdr_pit(y, pred)
            snr = self.snr_pit(y, pred)
            self.log("valid_sdr", sdr, on_step=True, on_epoch=True)
            self.log("valid_snr", snr, on_step=True, on_epoch=True)
        except:
            print("Failed to plot SDR and SNR")
            
        try:
            nb_pesq = self.nb_pesq(y,pred)
            self.log("valid_nb_pesq", nb_pesq, on_step=False, on_epoch=True)
            if self.wb_pesq is not None:
                wb_pesq = self.wb_pesq(y,pred)
                self.log("valid_wb_pesq", wb_pesq, on_step=False, on_epoch=True)
        except:
            print("Failed to plot PESQ")
        
        return loss
    def test_step(self, batch, batch_idx):
        _, x, y = batch
        y = torch.cat(y,dim = 1)
        
        pred = self.model(x)        
        
        
        si_sdr = self.si_sdr_pit(y, pred)
        si_snr = self.si_snr_pit(y, pred)
        loss = -si_snr
        self.log("loss", loss, on_step=True, on_epoch=True)
        self.log("sisdr", si_sdr, on_step=True, on_epoch=True)
        self.log("sisnr", si_snr, on_step=True, on_epoch=True)
        
        try:
            sdr = self.sdr_pit(y, pred)
            snr = self.snr_pit(y, pred)
            self.log("sdr", sdr, on_step=True, on_epoch=True)
            self.log("snr", snr, on_step=True, on_epoch=True)
        except:
            print("Failed to plot SDR and SNR")
        
        try:
            nb_pesq = self.nb_pesq(y,pred)
            self.log("nb_pesq", nb_pesq, on_step=False, on_epoch=True)
            if self.wb_pesq is not None:
                wb_pesq = self.wb_pesq(y,pred)
                self.log("wb_pesq", wb_pesq, on_step=False, on_epoch=True)
        except:
            print("Failed to plot PESQ")
        

    def configure_optimizers(self):
        optimizer = self.config['optimizer'](self.model.parameters())
        lr_scheduler = self.config['optimizer'].submodules[0](optimizer)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}    
    
    def setup(self,stage = None):
                
        self.si_sdr_pit =  PermutationInvariantTraining(
            torchmetrics.functional.scale_invariant_signal_distortion_ratio,
            eval_func='max'
        )
        self.si_snr_pit = PermutationInvariantTraining(
            torchmetrics.functional.scale_invariant_signal_noise_ratio,
            eval_func='max'
        )
        
        if stage in ['validate', 'test'] or stage is None:
            self.sdr_pit = PermutationInvariantTraining(
                torchmetrics.functional.signal_distortion_ratio,
                eval_func='max'
            )
            self.snr_pit = PermutationInvariantTraining(
                torchmetrics.functional.signal_noise_ratio,
                eval_func='max'
            )
            self.nb_pesq = pesq.PerceptualEvaluationSpeechQuality(self.config['sample_rate'], "nb")
            if self.config['sample_rate'] > 8000:
                self.wb_pesq = pesq.PerceptualEvaluationSpeechQuality(self.config['sample_rate'], "wb")
            else:
                self.wb_pesq = None
            
        if stage in ['fit', 'validate'] or stage is None:
            train_dataset = self.config.train_dataset()
            if hasattr(self.config, 'valid_dataset'):
                self.train_dataset = train_dataset
                self.valid_dataset = self.config.valid_dataset()
                print("VALIDATION DATASET LOADED!")
            else:
                def split_by_percentage(n:int, percent:float):
                    return [int(n * percent), n - int(n * percent)]

                self.train_dataset, self.valid_dataset = random_split(train_dataset, 
                    split_by_percentage(len(train_dataset), 0.95), generator=torch.Generator())
            
        if stage in ['test', 'predict'] or stage is None:
            self.test_dataset = self.config.test_dataset()
        return
        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=1, shuffle=True, num_workers=1)
    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=1, shuffle=False, num_workers=1)
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=1)