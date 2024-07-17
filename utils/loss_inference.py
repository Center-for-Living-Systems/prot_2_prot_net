import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class XY_loss_dict(nn.MSELoss):
    def __init__(self, **kwargs):
        print("LOSS KWARGS", kwargs)
        self.max_force= kwargs['loss_kwargs'].get('max_force')
        super().__init__()
        
    def forward(self, prediction, target, expweight=0.):
        mag = torch.linalg.norm(target, dim=1, keepdim=True) 
        
        MSE =F.mse_loss(prediction, target, reduction='none')

        loss_weight = torch.exp(torch.minimum(torch.abs(mag),self.max_force*torch.ones_like(mag))*expweight)
    
        return {'mse_loss': MSE.mean(), 'base_loss': torch.mean(MSE*loss_weight) }


class r_MSE_loss_dict(nn.MSELoss):
    def __init__(self, **kwargs):
        super().__init__()
    def forward(self, prediction, target, expweight=0.):
        MSE =F.mse_loss(prediction, target)
        return {'mse_loss': MSE, 'base_loss': MSE }

class weighted_MSE_loss_dict(nn.MSELoss):
    def __init__(self, **kwargs):
        super().__init__()
    def forward(self, prediction, target, expweight=0.):
        MSE =F.mse_loss(prediction, target)
        w_MSE_t = torch.tensordot(MSE, prediction)
        w_MSE = torch.tensordot(w_MSE_t, target)        
        return {'w_mse_loss': w_MSE, 'base_loss': MSE }

loss_function_dict = {                        
                        'xy': XY_loss_dict,
                        'r_mse': r_MSE_loss_dict,
                        'w_mse': weighted_MSE_loss_dict 
                        }
