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


class AM_loss_dict(nn.MSELoss): # Angle magnitude loss
    def __init__(self, **kwargs):
        print("LOSS KWARGS", kwargs)
        super().__init__()
        self.max_force= kwargs['loss_kwargs'].get('max_force')

    def forward(self, prediction, target, expweight=0., batch_avg=True):
        """
        MAG ERR, ANG ERR  HAVE SHAPES [B, H, W]
        """
        mag = target[..., -1, :, :]
        pred_mag = prediction[..., -1, :, :]

        nonzero = (mag>0)

        mag_err =F.mse_loss(pred_mag, mag, reduction='none')
        
        loss_weight = torch.exp(torch.minimum(torch.abs(mag),self.max_force*torch.ones_like(mag))*expweight)

        if batch_avg:
            mag_err_weighted = torch.mean(mag_err*loss_weight)
        else:
            mag_err_weighted = torch.mean(mag_err*loss_weight, axis=(-1,-2))
       
        mse_loss = (mag-pred_mag)**2 

        return {'base_loss': mag_err_weighted*10, 'mse_loss': mse_loss.mean().detach(), 'mag_loss': mag_err.mean().detach()}
    
    
    def all_metrics(self, prediction, target, mask, expweight=0., batch_avg=True):
        """
        MAG ERR, ANG ERR  HAVE SHAPES [B, H, W]
        """
        
        mag = target[..., -1, :, :]
        pred_mag = prediction[..., -2, :, :]

        nonzero = (mag>0)

        mag_err =F.mse_loss(pred_mag, mag, reduction='none')
        
        mag_err_l1 = torch.abs(mag - pred_mag)
        #print(mag_err.shape, mask.shape)
        
        if mask.shape[0]==1:
            # Just squeeze one dim
            mag_err[mask[0]==0] = torch.nan
        else:
            mag_err[mask.squeeze()==0] = torch.nan

        loss_weight = (mag-0.5)# *expweight

        mag_err_weighted = torch.nanmean(mag_err*loss_weight, axis=(-1,-2))

        if mask.shape[0]==1:
            # Just squeeze one dim
            mag[mask[0]==0] =torch.nan
            pred_mag[mask[0]==0] = torch.nan
        else:
            mag[mask.squeeze()==0] =torch.nan
            pred_mag[mask.squeeze()==0] = torch.nan

        mse_loss = (mag-pred_mag)**2*mag*pred_mag
        mse_mag_loss = torch.sqrt(mse_loss)
        
        mse_weighted = torch.nanmean(mse_loss*loss_weight, axis=(-1,-2))
        mse_mag_weighted = torch.nanmean(mse_mag_loss*loss_weight, axis=(-1,-2))
        
        #print(mse_weighted.shape, mse_loss.shape, loss_weight.shape, x.shape, mag.shape)
        
        
        #rmse_loss = torch.sqrt(mse_loss)
        #print('mag', mag.shape)
        #print('mag_err', mag_err.shape)
        #print('pred_mag', pred_mag.shape)

        return {'base_loss': mag_err_weighted*10 + ang_err_weighted, 
                    'mse_loss': torch.nanmean(mse_loss, axis=(-1,-2)).detach(), # <(vec F  - vec F')^2>
                    'mse_weighted': mse_weighted.detach(),                      # <(vec F  - vec F')^2>
                    'mse_mag_weighted': mse_mag_weighted.detach(), 
                    'mse_mag_loss': torch.nanmean(mse_mag_loss, axis=(-1,-2)).detach(), 
                    'mag_loss': torch.nanmean(mag_err, axis=(-1,-2)).detach(), # <(F  - F')^2>
                    'mag2_loss': torch.nanmean(torch.sqrt(mag_err), axis=(-1,-2)).detach(), # < |F  - F'| >
                    'mag_sum_loss': torch.nansum(mag_err, axis=(-1,-2)).detach(), # # < |F  - F'| >
                    'mag2_sum_loss': torch.nansum(torch.sqrt(mag_err), axis=(-1,-2)).detach(), 
                    'rel_mag_loss': torch.nanmean(mag_err/mag**2, axis=(-1,-2)).detach(), 
                    'rel2_mag_loss': torch.nanmean(mag_err/(0.5*(mag**2+pred_mag**2)), axis=(-1,-2)).detach(), 
                    'rel3_mag_loss': torch.nanmean(torch.sqrt(mag_err)/torch.sqrt(0.5*(mag**2+pred_mag**2)), axis=(-1,-2)).detach(),
                    'rel4_mag_loss': torch.nanmean(torch.sqrt(mag_err)/(0.5*(mag+pred_mag)), axis=(-1,-2)).detach(), 
                    'rel2_mse_loss': torch.nanmean(mse_loss/(0.5*(mag**2+pred_mag**2)), axis=(-1,-2)).detach(), 
                    'rel3_mse_loss': torch.nanmean(torch.sqrt(mse_loss)/torch.sqrt(0.5*(mag**2+pred_mag**2)), axis=(-1,-2)).detach(),
                    'rel4_mse_loss': torch.nanmean(torch.sqrt(mse_loss)/(0.5*(mag+pred_mag)), axis=(-1,-2)).detach(), 
                    'sum_F': torch.nansum(mag, axis=(-1,-2)).detach(), 
                    'sum_Fp': torch.nansum(pred_mag, axis=(-1,-2)).detach(), 
                    'mean_F': torch.nanmean(mag, axis=(-1,-2)).detach(), 
                    'mean_Fp': torch.nanmean(pred_mag, axis=(-1,-2)).detach(), 
                    'mean_F_Fp': torch.nanmean(torch.sqrt(0.5*(mag**2 + pred_mag**2)), axis=(-1,-2)).detach()}


loss_function_dict = {
                        'xy': XY_loss_dict,
                        'am': AM_loss_dict,
                        'r_mse': r_MSE_loss_dict, 
                        }
