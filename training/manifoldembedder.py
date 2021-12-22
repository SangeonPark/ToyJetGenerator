import sys
import os
sys.path.insert(0,'..')
sys.path.insert(0,'../optimal_transport')
from typing import Callable, Optional
from backbone import PositionalEncoding, particleTransformer
from emdloss import *
import torch.nn.functional as F
import pytorch_lightning as pl
from emdloss import *
import torch
import torch.nn.functional as F
from torch import nn
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional import accuracy
from pytorch_lightning.callbacks import Callback
from tqdm import tqdm

#PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64

class ManifoldEmbedder(pl.LightningModule):
    """docstring for ManifoldEmbedder"""
    def __init__(self, num_particles, embed_dim, learning_rate):
        super(ManifoldEmbedder, self).__init__()
        self.learning_rate = learning_rate
        self.num_particles = num_particles
        # particleTransformer takes inputs in the following order
        #(particle_feature_size, d_model, nhead, num_encoder_layers, num_decoder_layers, embed_dim, max_seq_length, pos_dropout, trans_dropout)
        self.encoder = particleTransformer(3, 8, 2, 2, 2, embed_dim, 8, .4, .4)

        #calculator for emd
        #self.emdcalc = EMDLoss(num_particles=num_particles,device='cuda')


    def forward(self, x):

        embedding = self.encoder(x)
        return embedding


    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y, dist = batch
        #x = x.view(x.size(0), -1)
        x_embed = self.encoder(x)
        y_embed = self.encoder(y)
        pdist = nn.PairwiseDistance(p=2)
        euclidean_dist = pdist(x_embed,y_embed)
        #print(euclidean_dist.size())
        #emd = self.emdcalc(x.reshape(-1,self.num_particles,3)[:,:,[1,2,0]], y.reshape(-1,self.num_particles,3)[:,:,[1,2,0]])

        loss = F.mse_loss(euclidean_dist, dist.float())
        # Logging to TensorBoard by default
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, dist = batch
        #x = x.view(x.size(0), -1)
        #print(x.size(), y.size(), dist.size())
        x_embed = self.encoder(x)
        y_embed = self.encoder(y)
        pdist = nn.PairwiseDistance(p=2)
        euclidean_dist = pdist(x_embed,y_embed)
        #emd = self.emdcalc(x, y)

        loss = F.mse_loss(euclidean_dist, dist.float())
        # Logging to TensorBoard by default
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, dist = batch
        #x = x.view(x.size(0), -1)
        #print(x.size(), y.size(), dist.size())
        x_embed = self.encoder(x)
        y_embed = self.encoder(y)
        pdist = nn.PairwiseDistance(p=2)
        euclidean_dist = pdist(x_embed,y_embed)
        #emd = self.emdcalc(x, y)

        loss = F.mse_loss(euclidean_dist, dist.float())
        # Logging to TensorBoard by default
        self.log("test_loss", loss, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        x, label = batch
        return self(x), label

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
        return optimizer

class JetDataset(torch.utils.data.Dataset):
    """It returns pair of jet data X, Y and the target emd(X,Y)"""
    def __init__(self, from_file, data_dir, jet1_data, jet2_data, num_part):
        super(JetDataset, self).__init__()
        if from_file:
            with open(os.path.join(data_dir,jet1_data), 'rb') as handle:
                self.jet1_data = pickle.load(handle)
            with open(os.path.join(data_dir,jet2_data), 'rb') as handle:
                self.jet2_data = pickle.load(handle)

        else:
            self.jet1_data = torch.FloatTensor(jet1_data.reshape(-1, num_part, 3))
            self.jet2_data = torch.FloatTensor(jet2_data.reshape(-1, num_part, 3))
        emdcalc = EMDLoss(num_particles=8,device='cpu')
        if torch.cuda.is_available():
            emdcalc = EMDLoss(num_particles=8,device='cuda')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        jet1_data = self.process_data(jet1_data, num_part, 3, False )
        jet2_data = self.process_data(jet2_data, num_part, 3, False )
        paired_data = torch.utils.data.TensorDataset(jet1_data, jet2_data)
        dataloader = DataLoader(paired_data, batch_size=128, shuffle=False)
        emd = torch.zeros(0)
        for x,y in tqdm(dataloader):
            emd = torch.cat((emd.to(device), emdcalc(x.to(device),y.to(device))))
        self.emd = emd.to("cpu").float()

    def process_data(self, data, num_part, num_feat, doNormalize):
        data = data.reshape(-1,num_part, num_feat)
        data = data[:,:,[1,2,0]]
        if doNormalize:
            data[:,:,2]/=np.sum(data[:,:,2],axis=1).reshape(-1,1)
        return torch.FloatTensor(data)

    def __len__(self):
        return len(self.emd)

    def __getitem__(self, index):
        return self.jet1_data[index], self.jet2_data[index], self.emd[index]

class JetPredictDataset(torch.utils.data.Dataset):
    """docstring for JetPredictDataset"""
    def __init__(self, from_file, data_dir, jet_data, label_data, num_part):
        super(JetPredictDataset, self).__init__()
        if from_file:
            with open(os.path.join(data_dir,jet_data), 'rb') as handle:
                self.jet_data = pickle.load(handle)

        else:
            self.jet_data = torch.FloatTensor(jet_data.reshape(-1, num_part, 3))

        self.label_data = torch.FloatTensor(label_data)

    def __len__(self):
        return len(self.label_data)

    def __getitem__(self, index):
        return self.jet_data[index], label_data[index]

class JetPairDataModule(LightningDataModule):
    def __init__(self, data_dir: str, file_dict, batch_size :int = BATCH_SIZE):
        super().__init__()
        self.num_types = len(file_dict)
        self.file_dict = file_dict
        self.data_dir = data_dir
        self.batch_size = batch_size

    # When doing distributed training, Datamodules have two optional arguments for
    # granular control over download/prepare/splitting data:

    # OPTIONAL, called only on 1 GPU/machine
    #def prepare_data(self):
        #MNIST(os.getcwd(), train=True, download=True)
        #MNIST(os.getcwd(), train=False, download=True)


    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage: Optional[str] = None):
        # transforms
        # split dataset
        if stage in (None, "fit"):
            self.jetpair_train = torch.load(os.path.join(self.data_dir,self.file_dict['train']))
            self.jetpair_val = torch.load(os.path.join(self.data_dir,self.file_dict['val']))
            #self.jetpair_train, self.jetpair_val = random_split(jetpair_train, [280000, 40000])
        if stage == "test":
            self.jetpair_test = torch.load(os.path.join(self.data_dir,self.file_dict['test']))
        if stage == "predict":
            self.jetpair_predict = torch.load(os.path.join(self.data_dir,self.file_dict['predict']))

    # return the dataloader for each split
    def train_dataloader(self):
        jetpair_train = DataLoader(self.jetpair_train, batch_size=self.batch_size)
        return jetpair_train

    def val_dataloader(self):
        jetpair_val = DataLoader(self.jetpair_val, batch_size=self.batch_size)
        return jetpair_val

    def test_dataloader(self):
        jetpair_test = DataLoader(self.jetpair_test, batch_size=self.batch_size)
        return jetpair_test

    def predict_dataloader(self):
        jetpair_predict = DataLoader(self.jetpair_predict, batch_size=self.batch_size)
        return jetpair_predict

class MNISTDataModule(LightningDataModule):
    def __init__(self, data_dir: str, file_names, batch_size :int = BATCH_SIZE):
        super().__init__()
        self.num_types = len(file_names)
        self.file_names = file_names
        self.data_dir = data_dir
        self.batch_size = batch_size

    # When doing distributed training, Datamodules have two optional arguments for
    # granular control over download/prepare/splitting data:

    # OPTIONAL, called only on 1 GPU/machine
    #def prepare_data(self):
        #MNIST(os.getcwd(), train=True, download=True)
        #MNIST(os.getcwd(), train=False, download=True)


    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage: Optional[str] = None):
        # transforms
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        # split dataset
        if stage in (None, "fit"):
            mnist_train = MNIST(os.getcwd(), train=True, transform=transform)
            self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])
        if stage == "validate":
            self.mnist_test = MNIST(os.getcwd(), train=False, transform=transform)
        if stage == "test":
            self.mnist_test = MNIST(os.getcwd(), train=False, transform=transform)
        if stage == "predict":
            self.mnist_predict = MNIST(os.getcwd(), train=False, transform=transform)

    # return the dataloader for each split
    def train_dataloader(self):
        mnist_train = DataLoader(self.mnist_train, batch_size=self.batch_size)
        return mnist_train

    def val_dataloader(self):
        mnist_val = DataLoader(self.mnist_val, batch_size=self.batch_size)
        return mnist_val

    def test_dataloader(self):
        mnist_test = DataLoader(self.mnist_test, batch_size=self.batch_size)
        return mnist_test

    def predict_dataloader(self):
        mnist_predict = DataLoader(self.mnist_predict, batch_size=self.batch_size)
        return mnist_predict





class PrintCallbacks(Callback):
    def on_init_start(self, trainer):
        print("Starting to init trainer!")

    def on_init_end(self, trainer):
        print("Trainer is init now")

    def on_train_end(self, trainer, pl_module):
        print("Training ended")






