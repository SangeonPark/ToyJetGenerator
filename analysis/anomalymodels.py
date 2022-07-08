from typing import Optional, Tuple

import os, sys
import pickle
import numpy as np


import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, Trainer


AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64

class AutoEncoder(LightningModule):
    """
    >>> LitAutoEncoder()  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    LitAutoEncoder(
      (encoder): ...
      (decoder): ...
    )
    """

    def __init__(self, hidden_dim: int = 64, lr:float = 1e-3, enc_layers = [500,500,100], dec_layers = [100,500,500], latent_dim = 3):
        super().__init__()
        self.save_hyperparameters()
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat,out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(num_features=out_feat))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(p=0.7, inplace = False))
            return layers
        
        #self.encoder = nn.Sequential(nn.Linear(48, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3))
        self.encoder = nn.Sequential(
                                     *block(48, enc_layers[0]),
                                     *[layers for i in range(len(enc_layers)-1) for layers in block(enc_layers[i],enc_layers[i+1])],
                                     nn.Linear(enc_layers[-1], latent_dim)
                                     )
        self.decoder = nn.Sequential(
                                     *block(latent_dim, dec_layers[0]),
                                     *[layers for i in range(len(dec_layers)-1) for layers in block(dec_layers[i],dec_layers[i+1])],
                                     nn.Linear(dec_layers[-1], 48)
                                     )
        #self.decoder = nn.Sequential(nn.Linear(3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 48))
        self.learning_rate = lr
        
    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = self._prepare_batch(batch)
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def _prepare_batch(self, batch):
        x, _ = batch
        return x.view(x.size(0), -1)

    def _common_step(self, batch, batch_idx, stage: str):
        x = self._prepare_batch(batch)
        loss = F.mse_loss(x, self(x))
        self.log(f"{stage}_loss", loss, on_step=True)
        return loss


class MLP(LightningModule):
  
    def __init__(self, layer_widths=[500,500,100]):
        super().__init__()
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat,out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(num_features=out_feat))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(p=0.7, inplace = False))
            return layers

        self.layers = nn.Sequential(
                                     *block(48, layer_widths[0]),
                                     *[layers for i in range(len(layer_widths)-1) for layers in block(layer_widths[i],layer_widths[i+1])],
                                     nn.Linear(layer_widths[-1], 1)
                                     )
        self.loss = nn.BCEWithLogitsLoss()
      
    def forward(self, x):
        return self.layers(x)
     
    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, _ = self._prepare_batch(batch)
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def _prepare_batch(self, batch):
        x, y = batch
        return x.view(x.size(0), -1), y

    def _common_step(self, batch, batch_idx, stage: str):
        x, y = self._prepare_batch(batch)
        loss = self.loss(torch.squeeze(self(x)),y)
        self.log(f"{stage}_loss", loss, on_step=True)
        return loss


class AnomalyDataModule(LightningDataModule):
    def __init__(self, file_dict, batch_size :int = BATCH_SIZE): 
        super().__init__()
        self.file_dict = file_dict
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        # transforms
        # split dataset
        if stage in (None, "fit"):
            self.jetpair_train = torch.load(os.path.join(self.file_dict['train']))
            self.jetpair_val = torch.load(os.path.join(self.file_dict['val']))
            #self.jetpair_train, self.jetpair_val = random_split(jetpair_train, [280000, 40000])
        if stage == "test":
            self.jetpair_test = torch.load(os.path.join(self.file_dict['test']))
        if stage == "predict":
            self.jetpair_predict = torch.load(os.path.join(self.file_dict['predict']))

    # return the dataloader for each split
    def train_dataloader(self):
        jetpair_train = DataLoader(self.jetpair_train, batch_size=self.batch_size,shuffle=True,num_workers=4)
        return jetpair_train

    def val_dataloader(self):
        jetpair_val = DataLoader(self.jetpair_val, batch_size=self.batch_size,num_workers=4)
        return jetpair_val

    def test_dataloader(self):
        jetpair_test = DataLoader(self.jetpair_test, batch_size=self.batch_size, shuffle=False)
        return jetpair_test

    def predict_dataloader(self):
        jetpair_predict = DataLoader(self.jetpair_predict, batch_size=self.batch_size, shuffle=False)
        return jetpair_predict

class SingleJetDataset(torch.utils.data.Dataset):
    """docstring for SingleJetDataset"""
    def __init__(self, from_file, data_dir, isToy, jet_data, label_data, num_part):
        super(SingleJetDataset, self).__init__()
        if from_file:
            with open(os.path.join(data_dir,jet_data), 'rb') as handle:
                self.jet_data = pickle.load(handle)

        else:
            self.jet_data = jet_data

        if isToy:
            self.jet_data = self.process_data(self.jet_data, num_part, 3, True)
        else:
            self.jet_data = self.process_jet_data_all(self.jet_data, num_part)

        
        self.label_data = torch.FloatTensor(label_data)
        
    def process_data(self, data, num_part, num_feat, doNormalize):
        data = data.reshape(-1,num_part, num_feat)
        data = data[:,:,[1,2,0]]
        if doNormalize:
            data[:,:,2]/=np.sum(data[:,:,2],axis=1).reshape(-1,1)
        return torch.FloatTensor(data)

    def process_jet_data_all(self, dt, num_part):
        def fix_phi(phi):
            phi %= (2*np.pi)
            if phi > np.pi:
                phi -= 2*np.pi
            return phi

        def rotate_eig(evt, num_part):
            new = np.copy(evt)
            cov_mat = np.cov(evt[:3*num_part].reshape(-1,3)[:, 1:3], aweights=evt[:3*num_part].reshape(-1,3)[:, 0] , rowvar=False)
            #print(new.shape)
            if np.isnan(np.sum(cov_mat)):
                return new
            eig_vals, eig_vecs = np.linalg.eig(cov_mat)
            #print(eig_vals)
            idx = eig_vals.argsort()[::1]   
            eig_vals = eig_vals[idx]
            eig_vecs = eig_vecs[:,idx]
            new[:3*num_part].reshape(-1,3)[:, 1:3] = np.matmul(evt[:3*num_part].reshape(-1,3)[:, 1:3], eig_vecs)
            #print(index)
            #index += 1
            #print(new.shape)
            return new

        def flip(evt, num_part):
            new = np.copy(evt)
            upper_quadrant = np.where(evt[:3*num_part].reshape(-1,3)[:,2]>0)
            lower_quadrant = np.where(evt[:3*num_part].reshape(-1,3)[:,2]<=0)
            upper_sum = np.sum(evt[:3*num_part].reshape(-1,3)[upper_quadrant,0])
            lower_sum = np.sum(evt[:3*num_part].reshape(-1,3)[lower_quadrant,0])
            if lower_sum > upper_sum:
                new[:3*num_part].reshape(-1,3)[:,2] *= -1
            return new

        def flip_eta(evt, num_part):
            new = np.copy(evt)
            right_quadrant = np.where(evt[:3*num_part].reshape(-1,3)[:,1]>0)
            left_quadrant = np.where(evt[:3*num_part].reshape(-1,3)[:,1]<=0)
            right_sum = np.sum(evt[:3*num_part].reshape(-1,3)[right_quadrant,0])
            left_sum = np.sum(evt[:3*num_part].reshape(-1,3)[left_quadrant,0])
            if left_sum > right_sum:
                new[:3*num_part].reshape(-1,3)[:,1] *= -1
            return new   

        temp = np.copy(dt)
        pt = temp[:,3*num_part]
        eta = temp[:,3*num_part+1]
        phi = temp[:,3*num_part+2]
        fix_phi_vec = np.vectorize(fix_phi)
        temp[:,:3*num_part].reshape(-1, num_part, 3)[:,:,0] /= pt.reshape(-1,1)
        #print(temp)
        #temp[:,:3*num_part].reshape(-1, num_part, 3)[:,:,0] /= np.sum(temp[:,:3*num_part].reshape(-1, 16, 3)[:,:,0] ,axis=1).reshape(-1,1)
        temp[:,:3*num_part].reshape(-1, num_part, 3)[:,:,1] -= eta.reshape(-1,1)
        temp[:,:3*num_part].reshape(-1, num_part, 3)[:,:,2] = fix_phi_vec(temp[:,:3*num_part].reshape(-1, num_part, 3)[:,:,2] - phi.reshape(-1,1) )
        temp2 = np.apply_along_axis(rotate_eig, 1, temp, num_part)
        temp3 = np.apply_along_axis(flip, 1, temp2, num_part)
        temp4 = np.apply_along_axis(flip_eta, 1, temp3, num_part)
        #temp2 = np.copy(temp)
        return torch.FloatTensor(temp4[:,:3*num_part].reshape(-1, num_part, 3)[:, :, [1, 2, 0]])

    def __len__(self):
        return len(self.label_data)

    def __getitem__(self, index):
        return self.jet_data[index], self.label_data[index]