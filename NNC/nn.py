#!/usr/bin/env python
# coding: utf-8

# Neural Network training and evaluation

import pandas as pd
import argparse
import os
import sys
import numpy as np
import random
import pickle
import time

import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append('utils')

import models               #model architecture
import train_eval           #NN train and evaluation
import misc                 #miscellaneous functions

from misc import print      #print function that displays time

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser("generate_tensors.py")

parser.add_argument("--train_dataset",                                help = "list of imgb batches used for training", type = str, default = None, required = False)
parser.add_argument("--test_dataset",                                 help = "list of imgb batches for evaluation/inference", type = str, default = None, required = False)
parser.add_argument("--output_dir",                                   help = "dir to save predictions and model/optimizer weights", type = str, default = 'predictions/', required = False)
parser.add_argument("--model_weight",                                 help = "initialization weight of the model", type = str, default = None, required = False)
parser.add_argument("--optimizer_weight",                             help = "initialization weight of the optimizer, use only to resume training", type = str, default = None, required = False)
parser.add_argument("--model_name",                                   help = "model name", type = str, default = 'ConvNN', required = False)
parser.add_argument("--seed",                                         help = "seed for neural network training", type = int, default = 0, required = False)
parser.add_argument("--tensor_width",                                 help = "tensor width", type = int, required = True)
parser.add_argument("--tensor_height",                                help = "tensor height, all variants with larger read depth will be cropped", type = int, required = True)
parser.add_argument("--max_depth",                                    help = "99th quantile of read depth  distribution", type = float, default = 150., required = False)
parser.add_argument("--val_fraction",                                 help = "fraction of train dataset to use for validation", type = float, default = 0, required = False)
parser.add_argument("--batch_size",                                   help = "number of imgb batches combined in one SGD batch at each SGD iteration", type = int, default = 1, required = False)
parser.add_argument("--learning_rate",                                help = "learning rate for optimizer", type = float, default = 1e-3, required = False)
parser.add_argument("--weight_decay",                                 help = "weight decay for optimizer", type = float, default = 0.1, required = False)
parser.add_argument("--dropout",                                      help = "dropout in fully connected layers", type = float, default = 0.5, required = False)
parser.add_argument("--tot_epochs",                                   help = "total number of training epochs", type = int, default = 20, required = False)
parser.add_argument("--lr_sch_milestones",                            help = "epoch at which the learning rate should be reduced", type = int, default = 15, required = False)
parser.add_argument("--lr_sch_gamma",                                 help = "learning rate reduction factor", type = float, default = 0.1, required = False)
parser.add_argument("--save_each",                                    help = "when to save model/optimizer weights, save_each=0 if saving is not needed, save_each=tot_epochs for results to be saved only at the end", type = int, default = 0, required = False)

input_params = vars(parser.parse_args())

input_params = misc.dotdict(input_params)

assert input_params.tensor_width>24, 'Minimal tensor width is 24'
assert input_params.tensor_height>10, 'Minimal tensor height is 10'

for param_name in ['output_dir', '\\',
'train_dataset', 'test_dataset', '\\',
'tensor_width', 'tensor_height', '\\',
'max_depth', '\\',
'val_fraction', '\\',
'tot_epochs', 'save_each', '\\',
'model_name', '\\',
'model_weight', 'optimizer_weight', '\\',
'seed', '\\',
'batch_size', 'learning_rate', 'weight_decay', 'dropout', '\\',
'lr_sch_milestones', 'lr_sch_gamma', '\\']:

    if param_name == '\\':
        print()
    else:
        print(f'{param_name.upper()}: {input_params[param_name]}')

#fix seed for initialization of neural network weights
random.seed(input_params.seed)
np.random.seed(input_params.seed)
torch.manual_seed(input_params.seed)

train_on, valid_on, test_on = 0, 0, 0 #will be set to 1 if the corresponding operation is expected

if input_params.train_dataset:

    train_valid_images = pd.read_csv(input_params.train_dataset, header=None).squeeze() #full path to imgb batches

    train_valid_images = train_valid_images.sample(frac=1., random_state=1).tolist() #shuffle

    N_valid = int(input_params.val_fraction*len(train_valid_images)) #numebr of validation instances

    valid_images, train_images = train_valid_images[:N_valid], train_valid_images[N_valid:]

    print(f'Train instances {len(train_images)}')

    print(f'Validation instances: {len(valid_images)}')

    train_on, valid_on = len(train_images)>0, len(valid_images)>0

if input_params.test_dataset:

    test_images = pd.read_csv(input_params.test_dataset, header=None).squeeze().tolist() #full path to imgb batches

    print(f'Test/Inference instances: {len(test_images)}')

    test_on = len(test_images)>0

assert train_on+valid_on+test_on>0, 'Insufficient number of instances for operation' #not enough tensors for training/evaluation

class TensorDataset(Dataset):

    '''
    Dataset of variant tensors
    '''

    def __init__(self,
                 data,           #full path to imgb batches with corresponding labels
                 target_height,  #target tensor height for the neural network
                 target_width,   #target tensor width for the neural network
                 max_depth,      #constant for normalizing read depth
                ):

        self.data = data
        self.target_height = target_height
        self.target_width = target_width
        self.max_depth = max_depth

        self.variant_meta = [[] for idx in range(len(self.data))]
        self.variant_misc_data = [[] for idx in range(len(self.data))]

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        '''
        Retrieve a single imgb batch with variant tensors

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

        imgb_path = self.data[idx] #retrieve imgb batch

        p_hot_correction_factor = 1e-4 #for p-hot reads encoded as ushort in variant_to_tensor function

        #load imgb batch of tensors
        with open(imgb_path, 'rb') as f:

            tensors = pickle.load(f)

        N_tensors = len(tensors['images']) #number of tensors in the current imgb batch

        if len(self.variant_meta[idx]) == 0:
            self.variant_meta[idx] = tensors['info'] #extract meta information about the variants in the batch: chrom, pos, ref, alt, vcf name
            self.variant_misc_data[idx] = misc.get_misc_tensor_data(self.variant_meta[idx], self.max_depth) #extract information added explicitly to fully connected layers (flanking regions)

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

        labels = [info["true_label"] for info in tensors['info']]

        tensors_dataset_idx = [(idx,x) for x in range(N_tensors)] # position of each tensor in the dataset (idx_of_imgb_batch, pos_in_imgb_batch), to keep track of each individual tensor

        misc_data = self.variant_misc_data[idx] #information to be added to fully connected layers

        return full_tensors, labels, misc_data, tensors_dataset_idx

def collate_fn(data):
    '''
    Collate imgb batches
    '''

    output = []

    for item in zip(*data):
        item_flattened = np.array([sample for batch in item for sample in batch])
        output.append(item_flattened)

    return output

#define train and evaluation datasets/dataloaders

if train_on:

    train_dataset = TensorDataset(train_images, target_height=input_params.tensor_height, target_width=input_params.tensor_width, max_depth=input_params.max_depth)

    train_dataloader = DataLoader(train_dataset, batch_size=input_params.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

if valid_on:

    valid_dataset = TensorDataset(valid_images, target_height=input_params.tensor_height, target_width=input_params.tensor_width, max_depth=input_params.max_depth)

    valid_dataloader = DataLoader(valid_dataset, batch_size=input_params.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

if test_on:

    test_dataset = TensorDataset(test_images, target_height=input_params.tensor_height, target_width=input_params.tensor_width, max_depth=input_params.max_depth)

    test_dataloader = DataLoader(test_dataset, batch_size=input_params.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

#access the GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('\nCUDA device: GPU\n')
else:
    device = torch.device('cpu')
    print('\nCUDA device: CPU\n')


if input_params.model_name=='ConvNN':
    model = models.ConvNN(dropout=input_params.dropout, target_width=input_params.tensor_width, target_height=input_params.tensor_height) #define model
elif input_params.model_name=='ConvNN_flanking':
    model = models.ConvNN_flanking(dropout=input_params.dropout, target_width=input_params.tensor_width, target_height=input_params.tensor_height) #define model

model = model.to(device) #model to CUDA

model_params = [p for p in model.parameters() if p.requires_grad] #model parameters for optimizer

#display the model architecture

#from torchsummary import summary
#summary(model,(14,150,150), batch_size=2)

optimizer = torch.optim.AdamW(model_params, lr=input_params.learning_rate, weight_decay=input_params.weight_decay) #define optimizer

last_epoch = 0

if input_params.model_weight:

    if torch.cuda.is_available():
        #load on gpu
        model.load_state_dict(torch.load(input_params.model_weight))
        if input_params.optimizer_weight:
            optimizer.load_state_dict(torch.load(input_params.optimizer_weight))
    else:
        #load on cpu
        model.load_state_dict(torch.load(input_params.model_weight, map_location=torch.device('cpu')))
        if input_params.optimizer_weight:
            optimizer.load_state_dict(torch.load(input_params.optimizer_weight, map_location=torch.device('cpu')))

    last_epoch = int(input_params.model_weight.split('_')[-3]) #infer previous epoch from input_params.model_weight

if train_on:
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[input_params.lr_sch_milestones],
                                                            gamma=input_params.lr_sch_gamma,
                                                            last_epoch=last_epoch-1, verbose=False) #define learning rate scheduler


predictions_dir = os.path.join(input_params.output_dir, 'predictions') #dir to save predictions
weights_dir = os.path.join(input_params.output_dir, 'weights') #dir to save model weights every save_each epoch

os.makedirs(predictions_dir, exist_ok = True)

if input_params.save_each:
    os.makedirs(weights_dir, exist_ok = True)

tot_epochs = max(last_epoch+1, input_params.tot_epochs)

tot_train_time, tot_test_time = 0, 0

train_pred, test_pred = [], []

for epoch in range(last_epoch+1, tot_epochs+1):

    if train_on:

        print(f'EPOCH {epoch}: Training...')

        start_time = time.time()

        train_loss, train_pred = train_eval.model_train(model, optimizer, train_dataloader, device)

        tot_train_time += time.time() - start_time

        lr_scheduler.step() #for MultiStepLR we take a step every epoch

        train_ROC_AUC, _ = misc.get_ROC(train_pred)

        print(f'EPOCH: {epoch} - train loss: {train_loss:.4}, train ROC AUC: {train_ROC_AUC:.4}')

        if input_params.save_each!=0 and (epoch%input_params.save_each==0 or epoch==tot_epochs): #save model weights

            misc.save_model_weights(model, optimizer, weights_dir, epoch)

            misc.save_predictions(train_pred, train_dataset, predictions_dir, f'training_epoch_{epoch}.vcf') #save train predictions on disk

    if valid_on:

        print(f'EPOCH {epoch}: Validating...')

        valid_loss, valid_pred = train_eval.model_eval(model, optimizer, valid_dataloader, device)

        valid_ROC_AUC, _ = misc.get_ROC(valid_pred)

        print(f'EPOCH: {epoch} - validation loss: {valid_loss:.4}, validation ROC AUC: {valid_ROC_AUC:.4}')

        misc.save_predictions(valid_pred, valid_dataset, predictions_dir, f'validation_epoch_{epoch}.vcf') #save validation predictions on disk

    if test_on and epoch==tot_epochs:

        print(f'EPOCH {epoch}: Test/Inference...')

        start_time = time.time()

        test_loss, test_pred = train_eval.model_eval(model, optimizer, test_dataloader, device)

        tot_test_time = time.time() - start_time

        _, _, labels = zip(*test_pred)

        if not None in labels: #if that's a test dataset

            test_ROC_AUC, _ = misc.get_ROC(test_pred)

            print(f'EPOCH: {epoch} - test loss: {test_loss:.4}, test ROC AUC: {test_ROC_AUC:.4}')

        misc.save_predictions(test_pred, test_dataset, predictions_dir, 'final_predictions.vcf') #save test/inference predictions on disk


print(f'peak memory allocation: {round(torch.cuda.max_memory_allocated(device)/1024/1024)} Mb')
print(f'total train time: {round(tot_train_time)} s : {round(len(train_pred)*(tot_epochs-last_epoch)/(tot_train_time+1))} samples/s')
print(f'total inference time: {round(tot_test_time)} s : {round(len(test_pred)/(tot_test_time+1))} samples/s')
print('Done')
