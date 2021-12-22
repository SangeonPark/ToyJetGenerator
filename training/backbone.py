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
    def __init__(self, particle_feature_size, d_model, nhead, num_encoder_layers, num_decoder_layers, embed_dim, max_seq_length, pos_dropout, trans_dropout):
        super().__init__()
        self.d_model = d_model
        self.embed_src = nn.Linear(particle_feature_size, d_model)
        self.embed_tgt = nn.Linear(particle_feature_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, pos_dropout, max_seq_length)

        #self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, trans_dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.NPART = max_seq_length

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat,out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(num_features=out_feat))
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(p=0.5, inplace = False))
            return layers


        layer_widths = [500,100,10]
        self.fcblock = nn.Sequential(
                                     *block(d_model*max_seq_length, layer_widths[0] ),
                                     *[layers for i in range(len(layer_widths)-1) for layers in block(layer_widths[i],layer_widths[i+1])],
                                     nn.Linear(layer_widths[-1], embed_dim)
                                     )
        #print(self.fcblock)



        #self.fc1 = nn.Linear(d_model*self.NPART, intermediate)
        #self.bn1 = nn.BatchNorm1d(num_features=intermediate)
        #self.activation1 = nn.LeakyReLU()
        #self.dropout1 = nn.Dropout(p=0.5, inplace=False)
        #self.fc2 = nn.Linear(intermediate, intermediate2)
        #self.bn2 = nn.BatchNorm1d(num_features=intermediate2)
        #self.activation2 = nn.LeakyReLU()
        #self.dropout2 = nn.Dropout(p=0.5, inplace=False)
        #self.fc3 = nn.Linear(intermediate2, intermediate3)
        #self.bn3 = nn.BatchNorm1d(num_features=intermediate3)
        #self.activation3 = nn.LeakyReLU()
        #self.dropout3 = nn.Dropout(p=0.5, inplace=False)
        #self.fc4 = nn.Linear(intermediate3, intermediate4)
        #self.bn4 = nn.BatchNorm1d(num_features=intermediate4)
        #self.activation4 = nn.LeakyReLU()
        #self.dropout4 = nn.Dropout(p=0.5, inplace=False)
        #self.final = nn.Linear(intermediate4, 1)

    def forward(self, src):

        src = src.permute(1,0,2)
        #tgt = tgt.permute(1,0,2)

        #src = self.embed_src(src)
        src = self.pos_enc(self.embed_src(src) * math.sqrt(self.d_model))
        #tgt = self.pos_enc(self.embed_tgt(tgt) * math.sqrt(self.d_model))
        #output = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask,
        #                          tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        output = self.transformer_encoder(src)
        output = output.reshape(-1,self.d_model*self.NPART)
        output = output.permute(1,0,2)
        #print(self.fcblock)
        output = self.fcblock(output)
        return output
