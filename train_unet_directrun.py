import argparse
from itertools import product as iterprod
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from argparse import ArgumentParser
import time
from datetime import datetime
from tqdm.autonotebook import tqdm

import utils.data_processing as dp
import matplotlib.pyplot as plt

from utils.UNeXt import UNet
from utils.loss import loss_function_dict

import utils.nb_utils as nb_utils

import pprint

np.random.seed(11) # for reproducibility
torch.manual_seed(11)


# %load_ext autoreload
# %autoreload 2
# %matplotlib inline



###############################################################################
import pandas as pd
print(torch.__version__)
print(np.__version__)
print(pd.__version__)
###############################################################################
batch_size = 8
num_workers = 8

# Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
pin_memory = True if torch.cuda.is_available() else False

# Dataset
directory  = '/net/projects/CLS/force_reference/data/ZyxAct_16kPa_small/'
test_cells = 'cell_1'


in_channels  = [[6,7]] # Example: [[4], [4,6], [4,6,7]]. Channel 4 is mask, 6 is zyxin, 7 is other protein (here actin)
#in_channels  = [[6],[7]] # Example: [[4], [4,6], [4,6,7]]. Channel 4 is mask, 6 is zyxin, 7 is other protein (here actin)
out_channels = (2,3) # (Fx, Fy)
transform_kwargs = {'crop_size': 512,
                    'output_channels': out_channels, 
                    'vector_components': [out_channels, (0,1)],
                    'magnitude_only': False,
                    'angmag': True,
                    'norm_output': {'rescale': 0.25, 'threshold': 0.4},
                    }

dataset_kwargs = { 
                    'root': directory,
                    'force_load': False,
                    'test_split': 'bycell',
                    'test_cells': test_cells,
                    'in_channels': in_channels, 
                    'out_channels': out_channels, 
                    'transform_kwargs': transform_kwargs,
                    'frames_to_keep': 256,
                    'input_baseline_normalization': 'outside_inside', # Comment on what these do
                    'output_baseline_normalization': 'mean_dataset',
                    'remake_dataset_csv': True,
                    'exclude_frames': [31,90]
                     }


dataset = dp.CellDataset( **dataset_kwargs )

train_loader = dataset.get_loader(dataset.train_indices, batch_size, num_workers, pin_memory)
validation_loader = dataset.get_loader(dataset.test_indices, batch_size, num_workers, pin_memory)
###############################################################################
df = dataset.info.copy()

df.head(10)
###############################################################################
# Print test cells: 
print(dataset.test_cells)
print(dataset.test_indices)
###############################################################################
fig,ax=plt.subplots(2,3,figsize=(2*3, 2*2), dpi=144)

cell = 'cell_1'
frame = 5

idx = dataset.info.index[(dataset.info.folder==cell)&(dataset.info.frame==frame)].tolist()[0] # Get index in dataframe that contains the right cell and frame.

sample = dataset[idx] # get item

print({key: sample[key].shape for key in sample.keys()})


ax[0][0].set_title('Mask')
ax[0][0].imshow(sample['mask'].squeeze(), origin='lower', cmap='gray', vmax=1, vmin=0)
ax[0][1].set_title('Zyxin')
ax[0][1].imshow(sample['zyxin'].squeeze(), origin='lower', cmap='gray', vmax=3, vmin=0)
ax[0][2].set_title('Actin')
ax[0][2].imshow(sample['actin'].squeeze(), origin='lower', cmap='gray', vmax=10, vmin=0)

ax[1][0].set_title('$|F|$')
ax[1][0].imshow(sample['output'].squeeze()[0], origin='lower', cmap='inferno')
ax[1][1].set_title('Force angle $\\alpha$')
ax[1][1].imshow(sample['output'].squeeze()[1], origin='lower', vmax=np.pi, vmin=-np.pi)
ax[1][2].set_title('$\\vec{F}$')
ax[1][2].imshow(sample['output'].squeeze()[0], origin='lower', cmap='inferno')
ax[1][2].quiver(*nb_utils.make_vector_field(*sample['output'].squeeze(), downsample=15, threshold=0.4, angmag=True), color='w', scale=20)

for a in ax.flat:
    a.set_xticks([])
    a.set_yticks([])
    
result_plot_dir = '/net/projects/CLS/force_reference/gitcodes/org_gitcode/result_plot_DSI'
plt.savefig(os.path.join(result_plot_dir,'example_training.png'))

###############################################################################
fig,ax=plt.subplots(1,1,figsize=(1, 1), dpi=200)

cell = 'cell_1'
frame = 5

idx = dataset.info.index[(dataset.info.folder==cell)&(dataset.info.frame==frame)].tolist()[0] # Get index in dataframe that contains the right cell and frame.

sample = dataset[idx] # get item


ax.imshow(sample['zyxin'].squeeze(), origin='lower', cmap='gray', vmax=10, vmin=0)

ax.set_xticks([])
ax.set_yticks([])


plt.savefig(os.path.join(result_plot_dir,'example_zyxin.png'))
        

###############################################################################
n_lyr  = 3 # number of downsampling layers
ds_krnl= 4 # downsample kernel
n_ch   = 4 # number of channels in the beginning of the network
n_blocks = 4 # number of ConvNext blocks, wherever ConvNext blocks are used

prepend_hparams = {'start_channel': 1, 'resnet_channel': n_ch, 'end_channel': n_ch, 'N_blocks': n_blocks,                                         # Args for architecture
                    'kernel': 7,'stride': 1, 'inv_bottleneck_factor': 4, 'dilation': 1,'dropout_rate': 0.1, 'activation': 'gelu', 'batchnorm': 1} # Args for ConvNext blocks
encoder_hparams = {'n_ch': n_ch, 'n_layers': n_lyr, 'N_node_blocks': n_blocks, 'N_skip_blocks': n_blocks,
                    'downsample_kwargs': {'kernel': ds_krnl, 'activation': 'gelu', 'batchnorm': 1},
                    'interlayer_kwargs': {'kernel': 7,'stride': 1, 'inv_bottleneck_factor': 4, 'dilation': 1,'dropout_rate': 0.1, 'activation': 'gelu', 'batchnorm': 1}
                    }
decoder_hparams = {'n_layers': n_lyr, 'N_node_blocks': n_blocks, 'upsample_kernel': ds_krnl,
                    'kernel': 7,'stride': 1, 'inv_bottleneck_factor': 4, 'dilation': 1,'dropout_rate': 0.1, 'activation': 'gelu', 'batchnorm': 1}
append_hparams = {'start_channel': n_ch, 'resnet_channel': n_ch, 'end_channel': 2, 'N_blocks': n_blocks,
                    'kernel': 7,'stride': 1, 'inv_bottleneck_factor': 8, 'dilation': 1,'dropout_rate': 0.1, 'activation': 'gelu', 'batchnorm': 1} 
optimizer_hparams = {'LR': 0.001, 'schedule_rate': 0.99}
loss_hparams = {'loss_type': 'am',
                'exp_weight': 0.0,
                'strainenergy_regularization': 0.0,
                'exp_schedule': {'type': 'linear', 'width': 310, 'e_crit': 30},
                'reg_schedule': {'type': 'linear', 'width': 310, 'e_crit': 30},
                'loss_kwargs': {'max_force': 8.}
               }





models = {}
for protein in ['zyxin', 'actin']:
    modelname = 'directrun_model_zyxin_actin_force_saveplot'

    logger_params = {'log_dir': f'./tensorboard_logs/{modelname}', 
                     'image_epoch_freq': 10,
                     'image_callbacks': 'vectorfield,hists',
                     'save_model_freq': 20}

    # Actually build model:
    model_kwargs={
                    'input_type':  protein, 
                    'prepend_hparams': prepend_hparams, 
                    'encoder_hparams': encoder_hparams, 
                    'decoder_hparams': decoder_hparams, 
                    'append_hparams': append_hparams, 
                    'optimizer_hparams': optimizer_hparams,
                    'loss_hparams': loss_hparams,
                    'logger_params': logger_params,
                    'name': 'model_0'}


    model = UNet( **model_kwargs, model_idx=0)
    model.to(device)
    
    models[protein] = model

###############################################################################
save_models = True
n_epochs = 150

pbar = tqdm(total=n_epochs*(np.minimum(dataset.frames_to_keep, len(dataset.train_indices))/batch_size))

t0 = time.time()
for e in range(n_epochs):
    pbar.set_description(f'Epoch {e}')
    loss_values_train = {}
    loss_values_val = {}

    for mkey in models.keys():
        models[mkey].reset_running_train_loss()
        models[mkey].reset_running_val_loss()

    for sample in train_loader: 
        for key in sample:
            sample[key] = sample[key].to(device)
        
        for mkey in models.keys():
            models[mkey].training_step(sample, epoch=e) # loss.backward() and optimizer step occur in here
        
        pbar.update(1)

    for sample in validation_loader:
        for key in sample:
            sample[key] = sample[key].to(device)
    
        for mkey in models.keys():
            models[mkey].validation_step(sample, epoch=e)

    for mkey in models.keys():
        models[mkey].scheduler.step()

    print("Epoch %u:\t Time: %0.2f \t(per epoch: %0.2f)"%(e, time.time()-t0, (time.time()-t0)/(e+1)))

    # SAVE
    if save_models:
        # Log in tensorboard
        # for mkey in models.keys():
        #     model[mkey].log_images(epoch=e)
        #     model[mkey].log_scalars(epoch=e) 
            
        # Save models
        if e%(logger_params['save_model_freq'])==0 or e==n_epochs-1: 
            torch.save({'model': model.state_dict(),
                        'model_kwargs': model_kwargs,
                        'model_name': model.name,
                        'model_idx': model.index,
                        'dataset_kwargs': dataset_kwargs,
                        'test_cells': dataset.test_cells,
                        }, 
                       os.path.join( model.logdir, 'model.pt') )

###############################################################################
fig,ax=plt.subplots(1,3,figsize=(3*3, 3*1), dpi=200)

model = models['actin']

eval_dataset_kwargs = dataset_kwargs
eval_dataset_kwargs['transform_kwargs']['crop_size'] = 960
eval_dataset_kwargs['exclude_frames'] = None
dataset_eval = dp.CellDataset( **eval_dataset_kwargs )

cell = 'cell_3'
frame = 0

idx = dataset_eval.info.index[(dataset_eval.info.folder==cell)&(dataset_eval.info.frame==frame)].tolist()[0] # Get index in dataframe that contains the right cell and frame.


sample = dataset_eval[idx] # get item

input_image = model.select_inputs(model.input_type, sample).unsqueeze(0).to(device)
print(input_image.shape)

pred = model(input_image).detach().cpu().numpy()

print(pred.shape)

ax[0].set_title('Zyxin')
ax[0].imshow(sample['zyxin'].squeeze(), origin='lower', cmap='gray', vmax=3, vmin=0)

ax[1].set_title('$\\vec{F}_{exp}$')
ax[1].imshow(sample['output'].squeeze()[0], origin='lower', cmap='inferno', vmax=4, vmin=0)
ax[1].quiver(*nb_utils.make_vector_field(*sample['output'].squeeze(), downsample=20, threshold=0.4, angmag=True), color='w', scale=20, width=0.003)

ax[2].set_title('$\\vec{F}_{NN}$')
ax[2].imshow(pred.squeeze()[0], origin='lower', cmap='inferno', vmax=4, vmin=0)
ax[2].quiver(*nb_utils.make_vector_field(*pred.squeeze(), downsample=20, threshold=0.4, angmag=True), color='w', scale=20, width=0.003)

for a in ax.flat:
    a.set_xticks([])
    a.set_yticks([])
    
    
plt.savefig(os.path.join(result_plot_dir,'example_pred_train.png'))
 
###############################################################################
fig,ax=plt.subplots(1,3,figsize=(3*3, 3*1), dpi=200)

model = models['actin']

eval_dataset_kwargs = dataset_kwargs
eval_dataset_kwargs['transform_kwargs']['crop_size'] = 960
eval_dataset_kwargs['exclude_frames'] = None
dataset_eval = dp.CellDataset( **eval_dataset_kwargs )

cell = 'cell_1'
frame = 100

idx = dataset_eval.info.index[(dataset_eval.info.folder==cell)&(dataset_eval.info.frame==frame)].tolist()[0] # Get index in dataframe that contains the right cell and frame.


sample = dataset_eval[idx] # get item

input_image = model.select_inputs(model.input_type, sample).unsqueeze(0).to(device)
print(input_image.shape)

pred = model(input_image).detach().cpu().numpy()

print(pred.shape)

ax[0].set_title('Zyxin')
ax[0].imshow(sample['zyxin'].squeeze(), origin='lower', cmap='gray', vmax=3, vmin=0)

ax[1].set_title('$\\vec{F}_{exp}$')
ax[1].imshow(sample['output'].squeeze()[0], origin='lower', cmap='inferno', vmax=4, vmin=0)
ax[1].quiver(*nb_utils.make_vector_field(*sample['output'].squeeze(), downsample=20, threshold=0.4, angmag=True), color='w', scale=20, width=0.003)

ax[2].set_title('$\\vec{F}_{NN}$')
ax[2].imshow(pred.squeeze()[0], origin='lower', cmap='inferno', vmax=4, vmin=0)
ax[2].quiver(*nb_utils.make_vector_field(*pred.squeeze(), downsample=20, threshold=0.4, angmag=True), color='w', scale=20, width=0.003)

for a in ax.flat:
    a.set_xticks([])
    a.set_yticks([])
    
plt.savefig(os.path.join(result_plot_dir,'example_pred_test.png'))

###############################################################################

###############################################################################

###############################################################################

###############################################################################

###############################################################################

###############################################################################

###############################################################################

###############################################################################

###############################################################################

###############################################################################

###############################################################################

###############################################################################

###############################################################################

###############################################################################

###############################################################################

###############################################################################

###############################################################################

###############################################################################

###############################################################################

###############################################################################

###############################################################################

###############################################################################

###############################################################################

###############################################################################

###############################################################################

###############################################################################

###############################################################################
