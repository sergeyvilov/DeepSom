#!/usr/bin/env python
# coding: utf-8

# Neural Network training and evaluation

import pandas as pd
import builtins
import time
import argparse
import os
import sys
import numpy as np
import pickle

import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append('utils')

import models               #model architecture
import train_eval           #NN train and evaluation
import misc                 #miscellaneous functions

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

class dotdict(dict):
    '''
    Dictionary with dot.notation access to attributes
    '''

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

parser = argparse.ArgumentParser("generate_tensors.py")

parser.add_argument("--negative_ds",                                  help = "full path to negative(artefacts+germline SNPs) imgb batches", type = str, default = '', required = False)
parser.add_argument("--positive_ds",                                  help = "full path to positive (somatic SNPs) imgb batches", type = str, default = '', required = False)
parser.add_argument("--output_dir",                                   help = "dir to save predictions and model/optimizer weights", type = str, default = 'predictions/', required = False)
parser.add_argument("--inference_mode",                               help = "perform inference on inference_ds", type = lambda x: bool(str2bool(x)), default = False, required = False)
parser.add_argument("--inference_ds",                                 help = "full path to inference imgb batches", type = str, default = '', required = False)
parser.add_argument("--load_weights",                                 help = "load NN and optimizer weights from a previous run", type = lambda x: bool(str2bool(x)), default = False, required = False)
parser.add_argument("--config_start_base",                            help = "config_start_base of the NN state to load, e.g. C:/checkpoints/epoch_20_weights", type = str, default = None, required = False)
parser.add_argument("--tensor_width",                                 help = "tensor width", type = int, required = True)
parser.add_argument("--tensor_height",                                help = "tensor height", type = int, required = True)
parser.add_argument("--val_fraction",                                 help = "fraction of train dataset to use for validation/evaluation", type = float, default = 0, required = False)
parser.add_argument("--batch_size",                                   help = "batch size at one SGD iteration", type = int, default = 8, required = False)
parser.add_argument("--learning_rate",                                help = "learning rate for optimizer", type = float, default = 1e-3, required = False)
parser.add_argument("--weight_decay",                                 help = "weight decay for optimizer", type = float, default = 0.1, required = False)
parser.add_argument("--tot_epochs",                                   help = "total number of training epochs", type = int, default = 20, required = False)
parser.add_argument("--lr_sch_milestones",                            help = "epoch at which the learning rate should be reduced", type = int, default = 15, required = False)
parser.add_argument("--lr_sch_gamma",                                 help = "learning rate reduction factor", type = float, default = 0.1, required = False)
parser.add_argument("--resample_train",                               help = "'upsample' to upsample the underrepresented class,'downsample' to downsample overrepresented class. Doesn't apply to validation/evaluation or inference", type = str, choices = ['upsample', 'downsample', 'None'], default = None, required = False)
parser.add_argument("--save_each",                                    help = "when to save model/optimizer parameters, save_each=0 if saving is not needed, save_each=num_epochs for results to be saved only at the end", type = int, default = 0, required = False)

input_params = vars(parser.parse_args())

input_params = dotdict(input_params)

assert input_params.tensor_width>24, 'Minimal tensor width is 24'
assert input_params.tensor_height>10, 'Minimal tensor height is 10'

for param_name in ['negative_ds', 'positive_ds', '\\',
'inference_mode', 'inference_ds', '\\',
'load_weights', 'config_start_base', '\\',
'val_fraction', '\\',
'output_dir', '\\',
'tensor_width','tensor_height', '\\',
'batch_size', 'learning_rate','weight_decay',  '\\',
'tot_epochs', 'lr_sch_milestones', 'lr_sch_gamma', '\\',
'resample_train',  '\\',
'save_each',  '\\']:
    if param_name == '\\':
        print()
    else:
        print(f'{param_name.upper()}: {input_params[param_name]}')

if not input_params.inference_mode:

    neg_df = pd.read_csv(input_params.negative_ds, names=['path'])
    neg_df['label'] = 0

    pos_df = pd.read_csv(input_params.positive_ds, names=['path'])
    pos_df['label'] = 1

    train_eval_df = pd.concat((neg_df, pos_df)) #concatenate positive and negative dataframes
    train_eval_df = train_eval_df.sample(frac=1, random_state=1) #shuffle

    N_eval_pos = int(input_params.val_fraction*len(pos_df)) #number of positive instances for evaluation
    N_eval_neg = int(input_params.val_fraction*len(neg_df)) #number of negative instances for evaluation

    eval_df = pd.concat((neg_df.sample(n=N_eval_neg, random_state=1), pos_df.sample(n=N_eval_pos, random_state=1))) #uniformly sample instanses for evaluation dataframe
    eval_df = eval_df.sample(frac=1, random_state=1) #shuffle

    train_df = pd.concat((train_eval_df, eval_df)).drop_duplicates(keep=False) #remove eval instances from train dataframe

    print(f'Train instances before resampling: {(train_df.label==1).sum()} positive; {(train_df.label==0).sum()} negative')

    train_df = misc.resample(train_df, resample_mode=input_params.resample_train)

    print(f'Train instances after resampling: {(train_df.label==1).sum()} positive; {(train_df.label==0).sum()} negative')

    print(f'Eval instances: {(eval_df.label==1).sum()} positive; {(eval_df.label==0).sum()} negative')

    train_enabled, eval_enabled = len(train_df)>0, len(eval_df)>0

else:

    eval_df = pd.read_csv(input_params.inference_ds, names=['path'])
    eval_df['label'] = None

    print(f'Performing inference on {len(eval_df)} instances')

    train_enabled, eval_enabled = False, True

assert train_enabled+eval_enabled>0, 'Insufficient number of instances for operation' #not enough tensors for training/evaluation

class TensorDataset(Dataset):

    '''
    Dataset of SNP tensors
    '''

    def __init__(self,
                 data,           #full path to imgb batches with corresponding labels
                 target_height,  #target tensor height for the neural network
                 target_width,   #target tensor width for the neural network
                ):

        self.data = data
        self.target_height = target_height
        self.target_width = target_width

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        '''
        Retrieve a tensor

        Input tensors are provided in imgb batches to reduce the time of disk I/O operations

        If tensor height is smaller than self.target_height,
        we pad it with 0 to reach the required self.target_height.
        If tensor height is larger that self.target_height,
        we remove some reads on the top and on the bottom, leaving the central part.

        If tensor width is smaller than self.target_width,
        we pad it with 0 to reach the required self.target_width.
        If tensor width is larger that self.target_width,
        we remove some reads on the left and on the right, leaving the central part.

        '''

        imgb_path, label = self.data[idx] #retrieve imgb batch

        p_hot_correction_factor = 1e-4 #for p-hot reads encoded as ushort in variant_to_tensor function

        #load imgb batch of tensors
        with open(imgb_path, 'rb') as f:

            tensors = pickle.load(f)

        N_tensors = len(tensors['images']) #number of tensors in the current imgb batch

        full_tensors = [] #tensors of right dimensions (self.target_height, self.target_width)

        #loop over all tensors in the imgb batch and adjust their size to (self.target_height, self.target_width)
        for tensor in tensors['images']:

            one_hot_ref = tensor['one_hot_ref']
            p_hot_reads = tensor['p_hot_reads']*p_hot_correction_factor
            flags_reads = tensor['flags_reads']

            tensor_height, tensor_width, _ = p_hot_reads.shape #current size

            one_hot_ref = np.tile(one_hot_ref, (tensor_height,1,1)) #propagate reference bases over all reads

            tensor = np.concatenate((one_hot_ref,p_hot_reads,flags_reads), axis=2)

            if self.target_height>tensor_height:

                #if there are not enough reads, pad tensor with 0 to reach the target_height
                padding_tensor = np.zeros((self.target_height-tensor_height, tensor_width, 14))
                full_tensor_h = np.concatenate((tensor, padding_tensor), axis = 0) #concatenate over the reads axis
                full_tensor_h = np.roll(full_tensor_h,max(self.target_height//2-tensor_height//2,0),axis=0) #put the piledup reads in the center

            else:

                #if there are too many reads, keep reads in the center, remove at the top and at the bottom
                shift = max(tensor_height//2-self.target_height//2,0)
                full_tensor_h = tensor[shift:shift+self.target_height,:,:]

            if self.target_width>tensor_width:

                #if reads are too short, pad reads with 0 to reach the target width
                padding_tensor = np.zeros((self.target_height, self.target_width-tensor_width, 14))
                full_tensor_w = np.concatenate((full_tensor_h, padding_tensor), axis = 1) #concatenate over the sequence axis
                full_tensor_w = np.roll(full_tensor_w,max(self.target_width//2-tensor_width//2,0),axis=1) #put the piledup reads in the center of tensor

            else:

                #if there are too many reads, keep reads in the center, remove on the left and on the right
                shift = max(tensor_width//2-self.target_width//2,0)
                full_tensor_w = full_tensor_h[:,shift:shift+self.target_width,:]

            full_tensor = np.transpose(full_tensor_w, (2,0,1)) #change dimensions order to CxWxH

            full_tensors.append(full_tensor)

        labels = [label]*len(full_tensors)

        tensors_dataset_idx = [(idx,x) for x in range(N_tensors)] # position of each tensor in the dataset (idx_of_imgb_batch, pos_in_imgb_batch), to keep track of each individual tensor

        return full_tensors, labels, tensors_dataset_idx

def collate_fn(data):
    '''
    Collate imgb batches
    '''

    output = []

    for item in zip(*data):
        item_flattened = [sample for batch in item for sample in batch]
        output.append(item_flattened)

    return output

#define train and evaluation datasets/dataloaders

if train_enabled:

    train_dataset = TensorDataset(train_df.values.tolist(), target_height=input_params.tensor_height, target_width=input_params.tensor_width)

    train_dataloader = DataLoader(train_dataset, batch_size=input_params.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

if eval_enabled:

    eval_dataset = TensorDataset(eval_df.values.tolist(), target_height=input_params.tensor_height, target_width=input_params.tensor_width)

    eval_dataloader = DataLoader(eval_dataset, batch_size=input_params.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

#access the GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('\nCUDA device: GPU\n')
else:
    device = torch.device('cpu')
    print('\nCUDA device: CPU\n')


model = models.ConvNN(dropout=0.5, target_width=input_params.tensor_width, target_height=input_params.tensor_height) #define model

model = model.to(device) #model to CUDA

model_params = [p for p in model.parameters() if p.requires_grad] #model parameters for optimizer

#display the model architecture

#from torchsummary import summary
#summary(model,(14,150,150), batch_size=2)

optimizer = torch.optim.AdamW(model_params, lr=input_params.learning_rate, weight_decay=input_params.weight_decay) #define optimizer

last_epoch = -1

if input_params.load_weights:

    if torch.cuda.is_available():
        #load on gpu
        model.load_state_dict(torch.load(input_params.config_start_base + '_model'))
        if not input_params.inference_mode:
            optimizer.load_state_dict(torch.load(input_params.config_start_base + '_optimizer'))
    else:
        #load on cpu
        model.load_state_dict(torch.load(input_params.config_start_base + '_model', map_location=torch.device('cpu')))
        if not input_params.inference_mode:
            optimizer.load_state_dict(torch.load(input_params.config_start_base + '_optimizer', map_location=torch.device('cpu')))

    last_epoch = int(input_params.config_start_base.split('_')[-2]) #infer previous epoch from input_params.config_start_base

if not input_params.inference_mode:
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[input_params.lr_sch_milestones],
                                                            gamma=input_params.lr_sch_gamma,
                                                            last_epoch=last_epoch, verbose=False) #define learning rate scheduler

#redefine print function for logging
def print(*args, **kwargs):
    now = time.strftime("[%Y/%m/%d-%H:%M:%S]-", time.localtime()) #place current time at the beggining of each printed line
    builtins.print(now, *args, **kwargs)
    sys.stdout.flush()

predictions_dir = os.path.join(input_params.output_dir, 'predictions') #dir to save predictions
weights_dir = os.path.join(input_params.output_dir, 'weights') #dir to save model weights every save_each epoch

os.makedirs(predictions_dir, exist_ok = True)
os.makedirs(weights_dir, exist_ok = True)

tot_epochs = max(last_epoch+2, input_params.tot_epochs)

for epoch in range(last_epoch+1, tot_epochs):

    if train_enabled:

        print(f'Training for epoch: {epoch}')

        train_loss, train_pred = train_eval.model_train(model, optimizer, train_dataloader, device)

        lr_scheduler.step() #for MultiStepLR we take a step every epoch

        train_ROC_AUC = misc.get_ROC(train_pred)

        print(f'EPOCH: {epoch} - train loss: {train_loss:.4}, train ROC AUC: {train_ROC_AUC:.4}')

        if input_params.save_each!=0 and (epoch+1)%input_params.save_each==0: #save model weights

            misc.save_model_weights(model, optimizer, weights_dir, epoch)

    if eval_enabled:

        print(f'Evaluating for epoch: {epoch}')

        eval_loss, eval_pred = train_eval.model_eval(model, optimizer, eval_dataloader, device, input_params.inference_mode)

        misc.save_predictions(eval_pred, eval_dataset, predictions_dir, epoch, input_params.inference_mode) #save evaluation predictions on disk

        if input_params.inference_mode:

            print(f'Inference completed. Predictions saved in {os.path.join(predictions_dir, "inference.csv")}')

            break

        eval_ROC_AUC = misc.get_ROC(eval_pred)

        print(f'EPOCH: {epoch} - eval loss:{eval_loss:.4}, eval ROC AUC: {eval_ROC_AUC:.4}')

        if not train_enabled:

            break

print('Done')
