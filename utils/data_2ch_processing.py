import os
import re
import torch
import numpy as np
import pandas as pd
import random

import scipy.ndimage
import skimage
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from skimage import filters
from skimage import morphology
from time import time
import tifffile

import warnings
warnings.filterwarnings("ignore")

'''
***************************************************

Datasets

***************************************************
'''

def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]

channel_to_protein_dict = {
                            '1': 'zyxin',
                            '2': 'pfak',
                            }


def transform_list(
                    output_channels=[],                    
                    crop_size=0,
                    norm_output={}, # keys: rescale, threshold. can also contain 'norm_to_max' bool
                    perturb_input={},
                    perturb_output={},
                    add_noise={}, # Sholud be dict with keys: 'type', 'other kwargs like std, N, R, if applicable' 
                    rotate=True
                    ):
    transform_list = []
    
    # if 'rescale' in perturb_input or 'rescale' in perturb_output: transform_list.append(RandomRescale(rescale_factor=0.7))
    # if 'rescale_deterministic' in perturb_input or 'rescale_deterministic' in perturb_output: transform_list.append(RescaleImage(rescale_factor=perturb_input['rescale_deterministic']))

    # transform_list.append(CellCrop(crop_size))

    # if 'smooth' in perturb_input: transform_list.append(SmoothForces())
    # if 'blur' in perturb_output: transform_list.append(RandomBlur()) # just blurs inputs
    # if add_noise: transform_list.append( AddNoise( **add_noise )) # Should be list with: type=[in_gaussian, out_gaussian, mag_peaks]
    # if norm_output: transform_list.append( Threshold(output_channels, **norm_output))  

    transform_list.append(ToTensor())

    transform = transforms.Compose(transform_list)

    return transform



def prediction_transforms(args, opt_args = {}):
    transform_list = []
    # transform_list.append(CellCrop(args.crop_size))
    # if 'zoom' in args.transform.split(','): 
    #     transform_list.append(ResolutionChange(args.zoomfactor))
        
    # if 'zyx_threshold' in opt_args: transform_list.append(BinarizeZyxin(opt_args['zyx_threshold']))
    # if 'zyx_scale' in opt_args: transform_list.append(ZyxinRescale(opt_args['zyx_scale']))
    # if 'peak_rescale' in opt_args: transform_list.append(PeakRescale(**opt_args['peak_rescale'])) # should be dict containing {n peaks: , rescale factor: }
    # if 'half_cut' in opt_args: transform_list.append(RandomHalfCut())
    # transform_list.append(EnforceSize(min_size=64))
    transform_list.append(ToTensor())
    # if 'smooth' in args.transform.split(','): transform_list.append(SmoothForces())
    
    # if normalize_output: Threshold(output_channels, **normalize_output)  

    print(transform_list)
    
    transform = transforms.Compose(transform_list)
    return transform


def args_to_transform_kwargs(norm_output=None, perturb_input='', perturb_output='', add_noise=''):
    n_o = {s.split(',')[0]: float( s.split(',')[1] ) for s in norm_output.split('/')} 
    p_i = perturb_input.split(',')
    p_o = perturb_output.split(',')
    try:
        noise = {s.split(',')[0]: float( s.split(',')[1] ) for s in add_noise.split('/')} 
    except: 
        print("Could not add noise to transform kwargs.")
        noise = {}
    return {'norm_output': n_o, 'perturb_input': p_i, 'perturb_output': p_o, 'add_noise': noise}

class CellDataset(Dataset):
    def __init__(self, 
                 root, 
                 force_load=False,
                 test_split='split_indices',
                 test_cells=None,
                 in_channels=[1],
                 out_channels=[2],
                 transform_kwargs={},
                 frames_to_keep=10000, # if <0, keeps all
                 input_baseline_normalization=None,
                 output_baseline_normalization=None,
                 validation_split=0.2,
                 remake_dataset_csv = False,
                 exclude_frames = None# [31,90]
                ):                       
        
        self.verbose = False
                        
        root = root.split(',')
        self.root = root

        self.transform = transform_list(**transform_kwargs) 
        print(transform_kwargs)
        self.validation_split = validation_split

        self.in_channels = in_channels
        self.in_unique = np.unique([int(x) for ch in self.in_channels for x in ch])
        self.out_channels = out_channels
       
        self.test_cells = test_cells

        self.frames_to_keep = frames_to_keep
        
        self.load_info(extensions=('tif'), force_remake=remake_dataset_csv, exclude_frames=exclude_frames)
        self.load_baselines(input_baseline_normalization)
        
        # Used below to go from frame -> integer time
        def get_time(row): return int(row.filename.split('.')[0].split('_')[-1])
                    
        self.split_indices() #validation_split=float(test_split) )

    
    def load_info(self, extensions, force_remake=False, exclude_frames=None):
        root = self.root
        csv_files = []
        for R in root:
            csv_file = os.path.join(R, 'dataset.csv')
            
            if os.path.exists(csv_file) or force_remake==False:
                if self.verbose: print('Dataset exists, path %s'%csv_file)
                csv_files.append(csv_file)
            else:
                print('csv doesn\'t exist, making...')
                filenames = []
                folders = []
                for subdir in os.listdir(R):
                    path = os.path.join(R, subdir)
                    if not os.path.isdir(path):
                        continue
                    count = 0
                    for name in sorted(os.listdir(path), key=natural_keys):
                        if name.lower().endswith(extensions):
                            filenames.append(name)
                            count += 1
                    folders += [subdir] * count
                self.info = pd.DataFrame({
                    'folder': folders,
                    'filename': filenames,
                })
                csv_files.append(csv_file)
                self.info['frame'] = self.info.filename.apply(lambda x: int(x.split('_')[-1].split('.')[0]) )
                self.info.to_csv(csv_file, index=False)

        infos = [pd.read_csv(csv) for csv in csv_files]

        for R, info in zip(root, infos):
            info['root'] = R
        
        self.info = pd.concat(infos, ignore_index=True)
        
        if exclude_frames is not None:
            #print("Excluding frames: ", exclude_frames)
            self.info = self.info[~self.info.frame.isin(np.arange(exclude_frames[0],exclude_frames[1]))].reset_index()

        return
    
    def load_baselines(self, remove_type):
        root = self.root
         
        if remove_type == 'totally_normalize': 
            if self.verbose: print("Normalizing inputs to 0, 1.")

            def rm_baseline_zyx(zyx_im, idx):
                zyx_im -= 92
                zyx_im /= (628 - 92) # to center around 0                
                return zyx_im

            def rm_baseline_act(act_im, idx):      
                act_im -= 99
                act_im /= (303-99) # to center around 0                
                return act_im        

        elif remove_type == 'none':
            def rm_baseline_zyx(zyx_im, idx): return zyx_im
            def rm_baseline_act(act_im, idx): return act_im
        else: 
            if self.verbose: print('Default inputbaseline removal')
            def rm_baseline_zyx(zyx_im, idx): return (zyx_im - zyx_im.min())/1000
            def rm_baseline_act(act_im, idx): return (act_im - act_im.min())/1000
            
        self.rm_baseline_zyx = rm_baseline_zyx
        self.rm_baseline_act = rm_baseline_act
        
        return

            
    def mask_crop(self, image, mask_idx=4, dilation_iter=50):
        dil = scipy.ndimage.binary_dilation(image[mask_idx], iterations=dilation_iter)#, structure=disk(r), iterations=1)
        image[:, dil==0] = 0
        return image

    def __len__(self):
        return len(self.info)


    def __getitem__(self, idx, mask_crop=True):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = tifffile.imread(os.path.join(self.info.root[idx], self.info.folder[idx], self.info.filename[idx]))

        image = self.transform(image)
        mask= image[2].unsqueeze(0)
        zyxin= image[0].unsqueeze(0)
        pfak = image[1].unsqueeze(0)

        output = image[self.out_channels].unsqueeze(0)
  
        return {'mask':image[2].unsqueeze(0),'zyxin': image[0].unsqueeze(0), 'pfak': image[1].unsqueeze(0), 'output': output}
    
            
    
    def get_loader(self, indices, batch_size, num_workers, pin_memory=True):
        sampler = SubsetRandomSampler(indices)
        loader = torch.utils.data.DataLoader(self, 
            batch_size=batch_size,
            num_workers=num_workers,
            sampler=sampler,
            pin_memory=pin_memory)
        return loader

    def get_loader_evaluation(self, indices, batch_size, num_workers, pin_memory=True):
        loader = torch.utils.data.DataLoader(self, 
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=pin_memory)
        return loader

    def split_indices(self, validation_split=0.2):
        indices = np.arange(len(self.info))
        if self.verbose: print('%u frames found'%len(indices))
        np.random.shuffle(indices)
        if self.frames_to_keep>0:
            indices = indices[:self.frames_to_keep]
        if self.verbose: print('%u frames kept'%len(indices))
        split   = int(np.floor(validation_split * len(indices)))
        self.train_indices, self.test_indices = indices[split:], indices[:split]
        
        self.test_cells = {'split': 'none'}

        return

    

class CellPredictDataset(CellDataset):
    def __init__(self, 
                 root=None,
                 subdir=None, 
                 csv_file=None, 
                 force_load=False,
                 in_channels=[1],
                 out_channels=(2),
                 multiinput=False,
                 opt_transform_args={},
                 remove_baseline='',
                 remove_force_baseline='',
                 swap_zyx_pax=False,
                 regions_given = 0):
        
        self.root = root.split(',')
        self.subdir = subdir
        self.transform = prediction_transforms(transform_args, opt_transform_args)
        self.remove_baseline = remove_baseline
        print("baseline: \t", remove_baseline)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.multiinput = multiinput
        self.swap_zyx_pax = swap_zyx_pax
        self.derivatives = derivatives
        self.regions_given = regions_given
        
        self.noise = False 

        self.load_info(extensions=('tif'))
        self.load_baselines(remove_baseline)
        self.load_force_baselines(remove_force_baseline)

        self.info = self.info.loc[self.info.folder == self.subdir]
        self.info.index = np.arange(len(self.info))
        
        self.loader = torch.utils.data.DataLoader(self, batch_size=1, pin_memory=False)
            
    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        inputs, outputs = super().__getitem__(idx)
        return inputs, outputs, os.path.join(self.root[0], str(self.info.folder[idx])), str(self.info.filename[idx])

        
class SubsetSampler(torch.utils.data.SequentialSampler):
    def __init__(self, indices) -> None:
        self.indices=indices # List

    def __iter__(self):
        return iter(self.indices)
    
#sampler = SubsetSampler(indices)
#loader = torch.utils.data.DataLoader(d, 
#    batch_size=1,
#    shuffle=False,
#    sampler=sampler,
#    pin_memory=True)


'''
***************************************************

Extra Image Processing Functions

***************************************************
'''

class RandomTranspose(object):
    def __init__(self, vector_components, prob=0.5):
        self.prob = prob
        self.vector_components = vector_components # was [2, 3]. Now is list of tuples [(0,1), (2,3) etc.] 

    def __call__(self, image):
        if np.random.random() < self.prob:
            image = np.swapaxes(image, -1, -2)

            for vec in self.vector_components:
                image[vec] = image[vec[::-1]]
        return image





class RandomFlip(object):
    def __init__(self, output_channels, prob=0.5, parity=None):
        """
        xy_dirs is the string of output channels
        """
        self.prob  = prob
        self.ndims = 2
        self.xy_dirs = output_channels[::-1] # Note they must be reversed!!!!!!!!! channels directions of x,y which need to be flipped
        print("RANDOM FLIP: ", self.xy_dirs)
        #print("RandomFlip: xy dirs (should be (ch2, ch1)\t", self.xy_dirs)

    def __call__(self, image):
        
        for dim in range(-self.ndims, 0):
            if np.random.random() < self.prob:
                image = np.flip(image, axis=dim)
                image[self.xy_dirs[dim]] *= -1

        return image

class CellCrop(object):
    """
    min_factor is the minimum factor that the image dimensions should be. I.e. if 16, then each dimension must be a multiple of 16
    """
    def __init__(self, end_size, pad_type='zeros'):
        self.end_size = end_size
        self.pad_type = pad_type
    
    def __call__(self, image, mask_channel=2):
        t = np.max(np.nonzero(image[mask_channel]), axis=1) # top right corner
        b = np.min(np.nonzero(image[mask_channel]), axis=1) # bottom left

        cell_size = t-b
        if np.any(cell_size>=self.end_size): 
            cell_size=np.ones_like(cell_size)*self.end_size # 960 max im size
            noname=1    
        if np.any(cell_size == self.end_size):
            noname2=1
            cent = np.round(((t+b)/2)).astype(int)
            flexible_range = np.floor(((t-b)-self.end_size)/2)
            flexible_range = np.maximum(flexible_range, 0)
            pad_amt = int(self.end_size/2)
            cent += int(self.end_size/2) 
            cent += [np.random.randint(-flexible_range[0], flexible_range[0]+1), np.random.randint(-flexible_range[1], flexible_range[1]+1)]
            
            pad = ((0,0), (pad_amt, pad_amt), (pad_amt,pad_amt))
            image = np.pad(image, pad, mode='constant', constant_values=0) # pad first so we don't run into boundary issues
            image = image[:,cent[0]-pad_amt:cent[0]+pad_amt,cent[1]-pad_amt:cent[1]+pad_amt]
        else:
            image = image[:, b[0]:t[0], b[1]:t[1]]

        pad = self.end_size - cell_size
        if np.any(pad<0): print(pad, cell_size)
        padL = np.floor(pad/2).astype(int)
        padR = np.ceil(pad/2).astype(int)

        pad = ((0,0),) + ((padL[0], padR[0]), (padL[1], padR[1]))
        
        if self.pad_type == 'zeros':
            image = np.pad(image, pad, mode='constant', constant_values=0.)
        elif self.pad_type == 'reflect':
            image = np.pad(image, pad, mode='reflect')

        try:
            assert(image.shape[-2:]==(self.end_size, self.end_size))
        except:
            print(image.shape)
            print(cell_size, t, b)
            print(np.unique(image[mask_channel]))
            print(noname, noname2)
            print(cent, pad_amt, pad)
        return image


class ToTensor(object):
    def __call__(self, image):
        return torch.tensor(image.copy(), dtype=torch.float32)

class ToNumpy(object):
    def __call__(self, image):
        return image.numpy()


class Threshold(object):
    def __init__(self, out_channels, threshold=0., rescale=1., norm_to_max=False):
        self.threshold = threshold
        self.rescale = rescale
        self.norm_to_max = norm_to_max
        
        self.out_channels = out_channels

    def __call__(self, image):
        im = image[self.out_channels, :, :]

    # Rescale
        if self.norm_to_max: im /= np.abs(im).max()
        else: im /= self.rescale
    # Threshold
        im[np.abs(im)<self.threshold] = 0
        image[self.out_channels, :, :] = im      
        return image


"""-----------------------------------------------------------

Dataset processing functions

Many of these random perturbations were never used, so may need to be debugged.

-----------------------------------------------------------"""
class ResolutionChange(object):
    def __init__(self, resolution_factor=1.1):
        self.res = resolution_factor

    def __call__(self, image):
        return scipy.ndimage.zoom(image, [1, self.res, self.res]) 

class RandomRescale(object):
    def __init__(self, rescale_factor=0.7):
        try:
            self.res = float(1)
        except: self.res=1
        if self.res>1: 
            self.res = 1/self.res

    def __call__(self, image):
        z = np.random.uniform(low=self.res, high=1/self.res)
        
        image = transforms.functional.affine(image, scale=z, angle=0, translate=[0,0], shear=0)
        return image

class RandomBlur(object):
    def __call__(self, image):
        if len(image)>=7:
            in_channels = [4,6,7]
        else: 
            in_channels = [4,6]
        
        kernel_size = np.random.randint(3)*2 + 1
        for ch in in_channels:
            chmax = torch.max(image[ch])
            image[ch] = transforms.GaussianBlur(kernel_size, sigma=(0.1, 5))(image[ch][None,...])
            image[ch] *= chmax/torch.max(image[ch])
        return image
    
class EnforceSize(object):
    def __init__(self, min_size = 64): # Minsize gives the factor which the image size must be a multiple of. I.e. image.shape = n*min_size (n \in Z)
        self.min_size = min_size
    
    def __call__(self, image):
        diff = 64 - np.mod(image.shape, 64)[-1]

        pads = np.asarray([np.floor(diff/2), np.ceil(diff/2)], dtype=int)
        pad_allax = [[0,0], pads, pads]

        padded = np.pad(image, pad_allax, mode='constant', constant_values=0)
        
        return padded
    
class ImageNorm(object):
    def __init__(self, inchannels):
        self.in_ch = inchannels
    def __call__(self, image):
        for ch in self.in_ch:
            image[ch] -= np.min(image[ch])
            image[ch] /= np.max(image[ch])       
        return image

class BinarizeZyxin(object):
    def __init__(self, threshold=0.3):
            
        self.threshold = threshold
        
    def __call__(self, image):
        if self.threshold=='mean':
            print('zyxin mean signal: ', np.mean(image[6]))
            image[6] = (image[6]>np.mean(image[6])).astype(float)
        else:
            image[6] = (image[6]>self.threshold*np.max(image[6])).astype(float)
        return image
    

class SmoothForces(object):
    def __init__(self, sigma=10, out_channels=[2,3]):
        self.outchs = out_channels
        self.sigma = sigma
    def __call__(self, image): #image has shape (8, H, W)
        for ch in self.outchs:
            immax = torch.max(image[ch])
            image[ch] = transforms.GaussianBlur(kernel_size=3, sigma=self.sigma)(image[ch][None, ...])
            image[ch] *= immax/torch.max(image[ch])
        assert len(image.shape)==3, 'Image shape '+str(image.shape)
        return image
    
class ZyxinRescale(object):
    def __init__(self, rescale_factor=2):
        self.rescale_factor = rescale_factor
        
    def __call__(self, image):
        image[0] *= self.rescale_factor
        return image
    
class RandomHalfCut(object):
    def __init__(self, rescale_factor=2):
        self.rescale_factor = rescale_factor
        
    def __call__(self, image):
        p = np.random.uniform()#0,1,2)
        print(p)
        H,W = image.shape[-2:]
        if p<0.25: # Left side 0
            image[0][:int(H/2),:] = 0
        if p>=0.25 and p<0.5:
            image[0][int(H/2):,:] = 0
        if p>=.5 and p<0.75: # Left side 0
            image[0][:,:int(W/2)] = 0
        if p>=0.75:
            image[0][:,int(W/2):] = 0
        
        return image
     
class RandomRotate(object):
    def __init__(self, vector_components):
        self.vec_chs = vector_components # List of lists
        
    def __call__(self, image):
        angle = np.random.uniform()*360
        image = scipy.ndimage.rotate(image, angle, axes=(-1,-2), reshape=False)
        angle *= np.pi/180
        
        for vc in self.vec_chs:
            "vc = [vx, vy] for each vector (pair of components) in the image"
            image[vc[1]], image[vc[0]] = image[vc[1]]*np.cos(angle) - image[vc[0]]*np.sin(angle), \
                                                image[vc[1]]*np.sin(angle) + image[vc[0]]*np.cos(angle)
        return image
    
class PeakRescale(object):
    def __init__(self, rescale_factor=3, numpeaks=1, peak_threshold=0.3):
        self.rescale_factor = rescale_factor
        self.numpeaks = numpeaks
        self.peak_threshold = peak_threshold
        
    def __call__(self, image):
        get_peak_regions(image[0], )
        image[6] *= self.rescale_factor
        return image
# end

class InputGaussianNoise(object):
    def __init__(self, std, input_channels, mask_channel=4):
        self.std = std
        self.inputs = input_channels
        self.mask = mask_channel

    def __call__(self, image):
        std = self.std*np.std(image[self.inputs])
        image[self.inputs] += np.random.randn(0, std)

        return image

class AddNoise(object):
    def __init__(self, noise_type, output_channels, mask_channel=4):
        if noisetype == 'in_gaussian': self.function = self.out_gaussian
        if noisetype == 'out_gaussian': self.function = self.out_gaussian
        if noisetype == 'force_scatter': self.function = self.force_scatter_noise

        self.fct_kwargs = kwargs
        self.output_channels = output_channels
        self.mask_channel = mask_channel

    def __call__(self, image):
        return self.function(image, **kwargs)


    def force_scatter_noise(self, image, N=0, R=10, intensity=1.):
        # Make mask to find nonzero entries
        mask = image[self.mask_channel]!=0
        nnz = np.nonzero(mask)
        
        centers = np.random.choice(len(nnz[0]), size=N)

        max_F = np.max( np.linalg.norm(image[self.output_channels], axis=0) )

        heights = np.random.uniform(0, max_F, size=N)*self.intensity
        sizes = np.random.uniform(0.5, 1.2, size=N) # This makes some peaks larger than others, although this is probably redundant. But multiplicative noise makes tails bigger

        noises = np.zeros_like( image[self.output_channels] ).astype(float)
        
        noises[:, nnz[0][centers], nnz[1][centers]] = heights*sizes

        noises[0] *= np.sin( image[self.output_channels[0]] )
        noises[1] *= np.cos( image[self.output_channels[1]] )

        noises[0] = scipy.ndimage.gaussian_filter(noises[0], sigma=R)*(2*np.pi)*R**2
        noises[1] = scipy.ndimage.gaussian_filter(noises[1], sigma=R)*(2*np.pi)*R**2

        image[self.output_channels[0]] += noises[0]
        image[self.output_channels[1]] += noises[1]

        return image


    def out_gaussian(self, image, std=0.):
        std = std*np.std(image[self.output_channels], axis=(-1, -2))
        image[self.output_channels] += np.random.randn(0, std)

        return image
        

class RescaleImage(object):
    def __init__(self, rescale_factor):
        self.F = rescale_factor

    def __call__(self, image):
        # Image has shape [8, H, W]
        #F = 0.108/0.175
        print("rescaling image by: ", self.F)

        image = skimage.transform.rescale(image, (1,self.F,self.F))
            
        return image 




