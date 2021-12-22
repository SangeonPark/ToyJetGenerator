import numpy as np
import torch

def process_data(data, num_part, num_feat, doNormalize):
    data = data.reshape(-1,num_part, num_feat)
    data[:,:,[0,1,2]] = data[:,:,[1,2,0]]
    if doNormalize:
        data[:,:,2]/=np.sum(data[:,:,2],axis=1).reshape(-1,1)
    return torch.Tensor(data)

