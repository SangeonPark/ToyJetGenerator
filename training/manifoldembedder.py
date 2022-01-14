import sys
import os
sys.path.insert(0,'..')
sys.path.insert(0,'../optimal_transport')
from typing import Callable, Optional
from backbone import PositionalEncoding, particleTransformer, MLP, RNN, CNN
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
#from tqdm.notebook import trange, tqdm
import pickle
from geomloss import SamplesLoss
import ot


#PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64

class ManifoldEmbedder(pl.LightningModule):
    """docstring for ManifoldEmbedder"""
    def __init__(self, data_type, data_npair, backbone_type, learning_rate, *args):
        super(ManifoldEmbedder, self).__init__()
        self.learning_rate = learning_rate
        self.data_npair = data_npair
        #self.num_particles = num_particles
        # particleTransformer takes inputs in the following order
        #(particle_feature_size, d_model, nhead, num_encoder_layers, num_decoder_layers, embed_dim, max_seq_length, pos_dropout, trans_dropout)
        if data_type == 'jets':
            if backbone_type == 'MLP':
                self.encoder = MLP(*args)
            elif backbone_type == 'Transformer':
                self.encoder = particleTransformer(3, *args)
            elif backbone_type == 'LSTM':
                self.encoder = RNN('LSTM',*args)
            elif backbone_type == 'GRU':
                self.encoder = RNN('GRU',*args)
            else:
                raise LookupError('only support MLP, Transformer, LSTM and GRU for jets')

        elif data_type == 'MNIST':
            if backbone_type == 'MLP':
                self.encoder = MLP(*args)

            elif backbone_type == 'CNN':
                self.encoder = CNN(*args)

            else:
                raise LookupError('only support MLP and CNN for MNIST')

        else:
            raise LookupError('only support Jets and MNIST dataembedding')
        #calculator for emd
        #self.emdcalc = EMDLoss(num_particles=num_particles,device='cuda')


    def forward(self, x):

        embedding = self.encoder(x)
        return embedding


    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        if self.data_npair == 2:
            x, y, dist = batch
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            dist = torch.where(dist > 0, dist, torch.tensor(0.01,dtype=torch.float).to(device))
            #x = x.view(x.size(0), -1)
            x_embed = self.encoder(x)
            y_embed = self.encoder(y)
            #print("xembed: ",x_embed[:2])
            #print("yembed: ",x_embed[:2])
            pdist = nn.PairwiseDistance(p=2)
            euclidean_dist = pdist(x_embed,y_embed)
            #print("euclidean: ", euclidean_dist[:2])
            #print("emd: ", dist[:2])
            #print(euclidean_dist.size())
            #emd = self.emdcalc(x.reshape(-1,self.num_particles,3)[:,:,[1,2,0]], y.reshape(-1,self.num_particles,3)[:,:,[1,2,0]])
            loss = torch.sum((euclidean_dist - dist.float()).abs() / (dist.float().abs() + 1e-8))/(len(euclidean_dist))

        elif self.data_npair == 3:
            x, y, z,  dist1, dist2, dist3 = batch
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            dist1 = torch.where(dist1 > 0, dist1, torch.tensor(0.01,dtype=torch.float).to(device))
            dist2 = torch.where(dist2 > 0, dist2, torch.tensor(0.01,dtype=torch.float).to(device))
            dist3 = torch.where(dist3 > 0, dist3, torch.tensor(0.01,dtype=torch.float).to(device))
            #x = x.view(x.size(0), -1)
            x_embed = self.encoder(x)
            y_embed = self.encoder(y)
            z_embed = self.encoder(z)
            #print("xembed: ",x_embed[0])
            #print("yembed: ",y_embed[0])
            #print("zembed: ",z_embed[0])
            pdist = nn.PairwiseDistance(p=2)
            euclidean_dist1 = pdist(x_embed,y_embed)
            euclidean_dist2 = pdist(y_embed,z_embed)
            euclidean_dist3 = pdist(z_embed,x_embed)
            #print(euclidean_dist1[:1],euclidean_dist2[:1],euclidean_dist3[:1])
            #print("euclidean: ", euclidean_dist[:2])
            #print("emd: ", dist[:2])
            #print(euclidean_dist.size())
            #emd = self.emdcalc(x.reshape(-1,self.num_particles,3)[:,:,[1,2,0]], y.reshape(-1,self.num_particles,3)[:,:,[1,2,0]])
            loss = (torch.sum((euclidean_dist1 - dist1.float()).abs() / (dist1.float().abs() + 1e-8)+(euclidean_dist2 - dist2.float()).abs() / (dist2.float().abs() + 1e-8)+(euclidean_dist3 - dist3.float()).abs() / (dist3.float().abs() + 1e-8))/(len(euclidean_dist1))) / 3.0


        #loss = F.mse_loss(euclidean_dist, dist.float())
        # Logging to TensorBoard by default
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.data_npair == 2:
            x, y, dist = batch
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            dist = torch.where(dist > 0, dist, torch.tensor(0.01,dtype=torch.float).to(device))
            #x = x.view(x.size(0), -1)
            x_embed = self.encoder(x)
            y_embed = self.encoder(y)
            #print("xembed: ",x_embed[:2])
            #print("yembed: ",x_embed[:2])
            pdist = nn.PairwiseDistance(p=2)
            euclidean_dist = pdist(x_embed,y_embed)
            #print("euclidean: ", euclidean_dist[:2])
            #print("emd: ", dist[:2])
            #print(euclidean_dist.size())
            #emd = self.emdcalc(x.reshape(-1,self.num_particles,3)[:,:,[1,2,0]], y.reshape(-1,self.num_particles,3)[:,:,[1,2,0]])
            loss = torch.sum((euclidean_dist - dist.float()).abs() / (dist.float().abs() + 1e-8))/(len(euclidean_dist))

        elif self.data_npair == 3:
            x, y, z,  dist1, dist2, dist3 = batch
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            dist1 = torch.where(dist1 > 0, dist1, torch.tensor(0.01,dtype=torch.float).to(device))
            dist2 = torch.where(dist2 > 0, dist2, torch.tensor(0.01,dtype=torch.float).to(device))
            dist3 = torch.where(dist3 > 0, dist3, torch.tensor(0.01,dtype=torch.float).to(device))
            #x = x.view(x.size(0), -1)
            x_embed = self.encoder(x)
            y_embed = self.encoder(y)
            z_embed = self.encoder(z)
            #print("xembed: ",x_embed[:2])
            #print("yembed: ",x_embed[:2])
            pdist = nn.PairwiseDistance(p=2)
            euclidean_dist1 = pdist(x_embed,y_embed)
            euclidean_dist2 = pdist(y_embed,z_embed)
            euclidean_dist3 = pdist(z_embed,x_embed)
            #print("euclidean: ", euclidean_dist[:2])
            #print("emd: ", dist[:2])
            #print(euclidean_dist.size())
            #emd = self.emdcalc(x.reshape(-1,self.num_particles,3)[:,:,[1,2,0]], y.reshape(-1,self.num_particles,3)[:,:,[1,2,0]])
            loss = (torch.sum((euclidean_dist1 - dist1.float()).abs() / (dist1.float().abs() + 1e-8)+(euclidean_dist2 - dist2.float()).abs() / (dist2.float().abs() + 1e-8)+(euclidean_dist3 - dist3.float()).abs() / (dist3.float().abs() + 1e-8))/(len(euclidean_dist1))) / 3.0

        # Logging to TensorBoard by default
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        if self.data_npair == 2:
            x, y, dist = batch
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            dist = torch.where(dist > 0, dist, torch.tensor(0.01,dtype=torch.float).to(device))
            #x = x.view(x.size(0), -1)
            x_embed = self.encoder(x)
            y_embed = self.encoder(y)
            #print("xembed: ",x_embed[:2])
            #print("yembed: ",x_embed[:2])
            pdist = nn.PairwiseDistance(p=2)
            euclidean_dist = pdist(x_embed,y_embed)
            #print("euclidean: ", euclidean_dist[:2])
            #print("emd: ", dist[:2])
            #print(euclidean_dist.size())
            #emd = self.emdcalc(x.reshape(-1,self.num_particles,3)[:,:,[1,2,0]], y.reshape(-1,self.num_particles,3)[:,:,[1,2,0]])
            loss = torch.sum((euclidean_dist - dist.float()).abs() / (dist.float().abs() + 1e-8))/(len(euclidean_dist))

        elif self.data_npair == 3:
            x, y, z,  dist1, dist2, dist3 = batch
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            dist1 = torch.where(dist1 > 0, dist1, torch.tensor(0.01,dtype=torch.float).to(device))
            dist2 = torch.where(dist2 > 0, dist2, torch.tensor(0.01,dtype=torch.float).to(device))
            dist3 = torch.where(dist3 > 0, dist3, torch.tensor(0.01,dtype=torch.float).to(device))
            #x = x.view(x.size(0), -1)
            x_embed = self.encoder(x)
            y_embed = self.encoder(y)
            z_embed = self.encoder(z)
            #print("xembed: ",x_embed[:2])
            #print("yembed: ",x_embed[:2])
            pdist = nn.PairwiseDistance(p=2)
            euclidean_dist1 = pdist(x_embed,y_embed)
            euclidean_dist2 = pdist(y_embed,z_embed)
            euclidean_dist3 = pdist(z_embed,x_embed)
            #print("euclidean: ", euclidean_dist[:2])
            #print("emd: ", dist[:2])
            #print(euclidean_dist.size())
            #emd = self.emdcalc(x.reshape(-1,self.num_particles,3)[:,:,[1,2,0]], y.reshape(-1,self.num_particles,3)[:,:,[1,2,0]])
            loss = (torch.sum((euclidean_dist1 - dist1.float()).abs() / (dist1.float().abs() + 1e-8)+(euclidean_dist2 - dist2.float()).abs() / (dist2.float().abs() + 1e-8)+(euclidean_dist3 - dist3.float()).abs() / (dist3.float().abs() + 1e-8))/(len(euclidean_dist1))) / 3.0

        self.log("test_loss", loss, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        x, label = batch
        #y[:,:,0] /= torch.max(y[:,:,0])
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
            self.jet1_data = jet1_data
            self.jet2_data = jet2_data
        #print(self.jet1_data[0],self.jet2_data[0])
        emdcalc = EMDLoss(num_particles=8,device='cpu')
        if torch.cuda.is_available():
            emdcalc = EMDLoss(num_particles=8,device='cuda')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.jet1_data = self.process_data(self.jet1_data, num_part, 3, True )
        self.jet2_data = self.process_data(self.jet2_data, num_part, 3, True )
        paired_data = torch.utils.data.TensorDataset(self.jet1_data, self.jet2_data)
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

class JetTripletDataset(torch.utils.data.Dataset):
    """It returns  of jet data X, Y, Z and the target emd(X,Y), emd(Y,Z), emd(Z,X)"""
    def __init__(self, from_file, data_dir, jet1_data, jet2_data, jet3_data, num_part):
        super(JetTripletDataset, self).__init__()
        if from_file:
            with open(os.path.join(data_dir,jet1_data), 'rb') as handle:
                self.jet1_data = pickle.load(handle)

            with open(os.path.join(data_dir,jet2_data), 'rb') as handle:
                self.jet2_data = pickle.load(handle)

            with open(os.path.join(data_dir,jet3_data), 'rb') as handle:
                self.jet3_data = pickle.load(handle)

        else:
            self.jet1_data = jet1_data
            self.jet2_data = jet2_data
            self.jet3_data = jet3_data
        #print(self.jet1_data[0],self.jet2_data[0])
        emdcalc = EMDLoss(num_particles=8,device='cpu')
        if torch.cuda.is_available():
            emdcalc = EMDLoss(num_particles=8,device='cuda')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.jet1_data = self.process_data(self.jet1_data, num_part, 3, True )
        self.jet2_data = self.process_data(self.jet2_data, num_part, 3, True )
        self.jet3_data = self.process_data(self.jet3_data, num_part, 3, True )
        paired_data_1 = torch.utils.data.TensorDataset(self.jet1_data, self.jet2_data)
        paired_data_2 = torch.utils.data.TensorDataset(self.jet2_data, self.jet3_data)
        paired_data_3 = torch.utils.data.TensorDataset(self.jet3_data, self.jet1_data)
        dataloader_1 = DataLoader(paired_data_1, batch_size=128, shuffle=False)
        dataloader_2 = DataLoader(paired_data_2, batch_size=128, shuffle=False)
        dataloader_3 = DataLoader(paired_data_3, batch_size=128, shuffle=False)
        emd_1 = torch.zeros(0)
        for x,y in tqdm(dataloader_1):
            emd_1 = torch.cat((emd_1.to(device), emdcalc(x.to(device),y.to(device))))

        emd_2 = torch.zeros(0)
        for x,y in tqdm(dataloader_2):
            emd_2 = torch.cat((emd_2.to(device), emdcalc(x.to(device),y.to(device))))


        emd_3 = torch.zeros(0)
        for x,y in tqdm(dataloader_3):
            emd_3 = torch.cat((emd_3.to(device), emdcalc(x.to(device),y.to(device))))



        self.emd_1 = emd_1.to("cpu").float()
        self.emd_2 = emd_2.to("cpu").float()
        self.emd_3 = emd_3.to("cpu").float()


    def process_data(self, data, num_part, num_feat, doNormalize):
        data = data.reshape(-1,num_part, num_feat)
        data = data[:,:,[1,2,0]]
        if doNormalize:
            data[:,:,2]/=np.sum(data[:,:,2],axis=1).reshape(-1,1)
        return torch.FloatTensor(data)

    def __len__(self):
        return len(self.emd_1)

    def __getitem__(self, index):
        return self.jet1_data[index], self.jet2_data[index], self.jet3_data[index], self.emd_1[index], self.emd_2[index] , self.emd_3[index]

class JetPredictDataset(torch.utils.data.Dataset):
    """docstring for JetPredictDataset"""
    def __init__(self, from_file, data_dir, jet_data, label_data, num_part):
        super(JetPredictDataset, self).__init__()
        if from_file:
            with open(os.path.join(data_dir,jet_data), 'rb') as handle:
                self.jet_data = pickle.load(handle)

        else:
            self.jet_data = jet_data

        self.jet_data = self.process_data(self.jet_data, num_part, 3, True )
        self.label_data = torch.FloatTensor(label_data)
        
    def process_data(self, data, num_part, num_feat, doNormalize):
        data = data.reshape(-1,num_part, num_feat)
        data = data[:,:,[1,2,0]]
        if doNormalize:
            data[:,:,2]/=np.sum(data[:,:,2],axis=1).reshape(-1,1)
        return torch.FloatTensor(data)

    def __len__(self):
        return len(self.label_data)

    def __getitem__(self, index):
        return self.jet_data[index], self.label_data[index]

class MNISTPredictDataset(torch.utils.data.Dataset):
    """docstring for MNISTPredictDataset"""
    def __init__(self, from_file, data_dir, MNIST_data, label_data):
        super(MNISTPredictDataset, self).__init__()
        if from_file:
            with open(os.path.join(data_dir,MNIST_data), 'rb') as handle:
                self.MNIST_data = pickle.load(handle)

        else:
            self.MNIST_data = torch.FloatTensor(MNIST_data)

        self.label_data = label_data

    def __len__(self):
        return len(self.label_data)

    def __getitem__(self, index):
        return self.MNIST_data[index], self.label_data[index]

class JetDataModule(LightningDataModule):
    def __init__(self, file_dict, batch_size :int = BATCH_SIZE):
        super().__init__()
        #self.num_types = len(file_dict)
        self.file_dict = file_dict
        #self.data_dir = data_dir
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
        jetpair_test = DataLoader(self.jetpair_test, batch_size=self.batch_size)
        return jetpair_test

    def predict_dataloader(self):
        jetpair_predict = DataLoader(self.jetpair_predict, batch_size=self.batch_size)
        return jetpair_predict


class MNISTDataset(torch.utils.data.Dataset):
    """It returns pair of jet data X, Y and the target emd(X,Y)"""
    def __init__(self, ot_method, from_file, data_dir, digit1_data, digit2_data):
        super(MNISTDataset, self).__init__()
        if from_file:
            with open(os.path.join(data_dir,digit1_data), 'rb') as handle:
                self.digit1_data = pickle.load(handle)

            with open(os.path.join(data_dir,digit2_data), 'rb') as handle:
                self.digit2_data = pickle.load(handle)

        else:
            self.digit1_data = torch.FloatTensor(digit1_data)
            self.digit2_data = torch.FloatTensor(digit2_data)

        if ot_method == 'sinkhorn':
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
            #loss(norm_image.view(28,28).to(device),mean_norm_image.to(device)).data.cpu().numpy()
            #emdcalc = EMDLoss(num_particles=8,device='cpu')
            #if torch.cuda.is_available():
            #    emdcalc = EMDLoss(num_particles=8,device='cuda')
            #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            #self.digit1_data = self.process_data(self.digit1_data, num_part, 3, False )
            #self.digit2_data = self.process_data(self.digit2_data, num_part, 3, False )
            paired_data = torch.utils.data.TensorDataset(self.digit1_data, self.digit2_data)
            dataloader = DataLoader(paired_data, batch_size=1, shuffle=False)
            OT = torch.zeros(len(self.digit1_data))
            for i,(x,y) in enumerate(dataloader):
                if (i % 100) == 0:
                    print(f"iteration {i}")
                #print(loss(x.view(28,28).to(device),y.view(28,28).to(device)))
                OT[i] = loss(x.view(28,28).to(device),y.view(28,28).to(device))
            self.OT = OT.to("cpu").float()

        elif ot_method == 'POT':
            x,y = np.indices((28,28))
            xs = np.zeros((28*28, 2))
            xt = np.zeros((28*28, 2))
            xs[:,0] = x.reshape(-1)
            xs[:,1] = y.reshape(-1)
            xt[:,0] = x.reshape(-1)
            xt[:,1] = y.reshape(-1)
            M = ot.dist(xs, xt, metric='euclidean')
            M /= M.max()

            paired_data = torch.utils.data.TensorDataset(self.digit1_data, self.digit2_data)
            dataloader = DataLoader(paired_data, batch_size=1, shuffle=False)
            OT = np.zeros(len(self.digit1_data))
            for i,(x,y) in enumerate(dataloader):
                if (i % 100) == 0:
                    print(f"iteration {i}")
                #print(loss(x.view(28,28).to(device),y.view(28,28).to(device)))
                x = x.view(28,28)
                y = y.view(28,28)
                x = x.data.numpy().astype(np.float64)
                y = y.data.numpy().astype(np.float64)
                x /= np.sum(x)
                y /= np.sum(y)
                #print(x)
                OT[i] = ot.emd2(x.reshape(-1), y.reshape(-1), M)
                #OT[i] = ot.emd2(x, y, M)
            self.OT = torch.FloatTensor(OT)


    #def process_data(self, data, num_part, num_feat, doNormalize):
    #    data = data.reshape(-1,num_part, num_feat)
    #    data = data[:,:,[1,2,0]]
    #    if doNormalize:
    #        data[:,:,2]/=np.sum(data[:,:,2],axis=1).reshape(-1,1)
    #    return torch.FloatTensor(data)

    def __len__(self):
        return len(self.OT)

    def __getitem__(self, index):
        return self.digit1_data[index], self.digit2_data[index], self.OT[index]


class MNISTPairDataModule(LightningDataModule):
    def __init__(self, file_dict, batch_size :int = BATCH_SIZE):
        super().__init__()
        self.file_dict = file_dict
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
            train_list = []
            for file in self.file_dict['train']:
                train_list.append(torch.load(file))
            self.mnist_train = torch.utils.data.ConcatDataset(train_list)

            val_list = []
            for file in self.file_dict['val']:
                val_list.append(torch.load(file))
            self.mnist_val = torch.utils.data.ConcatDataset(val_list)

        if stage == "test":
            test_list = []
            for file in self.file_dict['test']:
                test_list.append(torch.load(file))
            self.mnist_test = torch.utils.data.ConcatDataset(test_list)

        if stage == "predict":
            predict_list = []
            for file in self.file_dict['predict']:
                predict_list.append(torch.load(file))
            self.mnist_predict = torch.utils.data.ConcatDataset(predict_list)
    # return the dataloader for each split
    def train_dataloader(self):
        mnist_train = DataLoader(self.mnist_train, batch_size=self.batch_size,shuffle=True,num_workers=4)
        return mnist_train

    def val_dataloader(self):
        mnist_val = DataLoader(self.mnist_val, batch_size=self.batch_size,num_workers=4)
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







