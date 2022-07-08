import torch
import math
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)


    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class particleTransformer(nn.Module):
    def __init__(self, particle_feature_size, d_model, nhead, num_encoder_layers, num_decoder_layers, embed_dim, max_seq_length, pos_dropout, trans_dropout, layer_widths):
        super().__init__()  
        self.d_model = d_model
        self.embed_src = nn.Linear(particle_feature_size, d_model)
        self.embed_tgt = nn.Linear(particle_feature_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, pos_dropout, max_seq_length)

        #self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, trans_dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.NPART = max_seq_length

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat,out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(num_features=out_feat))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(p=0.5, inplace = False))
            return layers


        #layer_widths = [200,50,10]
        self.fcblock = nn.Sequential(
                                     *block(d_model*max_seq_length, layer_widths[0] ),
                                     *[layers for i in range(len(layer_widths)-1) for layers in block(layer_widths[i],layer_widths[i+1])],
                                     nn.Linear(layer_widths[-1], embed_dim)
                                     )
        #print(self.fcblock)


    def forward(self, src):

        src = src.permute(1,0,2)
        #tgt = tgt.permute(1,0,2)

        #src = self.embed_src(src)
        src = self.pos_enc(self.embed_src(src) * math.sqrt(self.d_model))
        #tgt = self.pos_enc(self.embed_tgt(tgt) * math.sqrt(self.d_model))
        #output = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask,
        #                          tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        output = self.transformer_encoder(src)
        output = output.permute(1,0,2)
        output = output.reshape(-1,self.d_model*self.NPART)
        #print(self.fcblock)
        output = self.fcblock(output)
        return output


class MLP(nn.Module):
    """docstring for MLP"""
    def __init__(self, input_dim, embed_dim, layer_widths):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat,out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(num_features=out_feat))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(p=0.7, inplace = False))
            return layers

        self.fcblock = nn.Sequential(
                                     *block(input_dim, layer_widths[0] ),
                                     *[layers for i in range(len(layer_widths)-1) for layers in block(layer_widths[i],layer_widths[i+1])],
                                     nn.Linear(layer_widths[-1], embed_dim)
                                     )
    def forward(self, src):
        output = src.reshape(-1,self.input_dim)
        output = self.fcblock(output)
        return output

class RNN(nn.Module):
    """docstring for MLP"""
    def __init__(self, rnn_model, num_particle, embed_dim, hidden_size, num_rnn_layers, layer_widths):
        super(RNN, self).__init__()
        self.num_particle = num_particle
        self.hidden_size = hidden_size
        #self.use_last = use_last

        if rnn_model == 'LSTM':
            self.rnn = nn.LSTM( input_size=3, hidden_size=hidden_size, num_layers=num_rnn_layers, dropout=0.5,
                                batch_first=True, bidirectional=True)
        elif rnn_model == 'GRU':
            self.rnn = nn.GRU( input_size=3, hidden_size=hidden_size, num_layers=num_rnn_layers, dropout=0.5,
                                batch_first=True, bidirectional=True)
        else:
            raise LookupError(' only support LSTM and GRU')

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat,out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(num_features=out_feat))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(p=0.6, inplace = False))
            return layers

        self.fcblock = nn.Sequential(
                                     nn.BatchNorm1d(hidden_size*2),
                                     *block(hidden_size*2, layer_widths[0] ),
                                     *[layers for i in range(len(layer_widths)-1) for layers in block(layer_widths[i],layer_widths[i+1])],
                                     nn.Linear(layer_widths[-1], embed_dim)
                                     )

    def forward(self, src):
        #output = src.reshape(-1,3*self.num_particle)
        output, ht = self.rnn(src, None)
        output = output[:,-1,:]
        output = output.reshape(-1, 2 * self.hidden_size)
        output = self.fcblock(output)
        return output


class CNN(nn.Module):
    def __init__(self, embed_dim, layer_widths):
        super(CNN, self).__init__()
        self.embed_dim = embed_dim
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat,out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(num_features=out_feat))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(p=0.4, inplace = False))
            return layers

        self.convblock = nn.Sequential(
                                       nn.Conv2d(1, 32, kernel_size=5),
                                       nn.ReLU(),
                                       nn.MaxPool2d(kernel_size=2),
                                       nn.Conv2d(32, 32, kernel_size=5),
                                       nn.ReLU(),
                                       #nn.MaxPool2d(2),
                                       nn.Conv2d(32,64, kernel_size=2),
                                       nn.ReLU(),
                                       nn.MaxPool2d(2),
                                       )


        self.fcblock = nn.Sequential(
                                     *block(3*3*64, layer_widths[0] ),
                                     *[layers for i in range(len(layer_widths)-1) for layers in block(layer_widths[i],layer_widths[i+1])],
                                     nn.Linear(layer_widths[-1], embed_dim)
                                     )




    def forward(self, x):
        x = x.view(-1, 1, 28,28)
        x = self.convblock(x)
        x = x.view(-1,3*3*64)
        x = self.fcblock(x)
        return x
