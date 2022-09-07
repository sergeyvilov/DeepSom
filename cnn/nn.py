#!/usr/bin/env python
# coding: utf-8

# CNN training and evaluation

import pandas as pd
import argparse
import os
import sys
import numpy as np
import random
import pickle
import time
from itertools import chain

import torch
from torch.utils.data import IterableDataset, DataLoader

import utils.models as models              #model architecture
import utils.train_eval as train_eval          #CNN train and evaluation
import utils.misc as misc                #miscellaneous functions

from utils.misc import print      #print function that displays time

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser("nn.py")

parser.add_argument("--train_dataset",                                help = "list of imgb batches used for training", type = str, default = None, required = False)
parser.add_argument("--test_dataset",                                 help = "list of imgb batches for evaluation/inference", type = str, default = None, required = False)
parser.add_argument("--output_dir",                                   help = "dir to save predictions and model/optimizer weights", type = str, default = '', required = False)
parser.add_argument("--model_weight",                                 help = "initialization weight of the model", type = str, default = None, required = False)
parser.add_argument("--optimizer_weight",                             help = "initialization weight of the optimizer, use only to resume training", type = str, default = None, required = False)
parser.add_argument("--seed",                                         help = "seed for neural network training", type = int, default = 0, required = False)
parser.add_argument("--tensor_width",                                 help = "tensor width", type = int, required = True)
parser.add_argument("--tensor_height",                                help = "tensor height, all variants with larger read depth will be cropped", type = int, required = True)
parser.add_argument("--max_depth",                                    help = "99th quantile of read depth  distribution", type = float, default = 200., required = False)
parser.add_argument("--max_train_tensors",                            help = "Maximal number of train tensors", type = int, default = None, required = False)
parser.add_argument("--max_valid_tensors",                            help = "Maximal number of validation tensors", type = int, default = None, required = False)
parser.add_argument("--max_test_tensors",                             help = "Maximal number of test tensors", type = int, default = None, required = False)
parser.add_argument("--val_fraction",                                 help = "fraction of train imgb batches to use for validation", type = float, default = 0, required = False)
parser.add_argument("--batch_size",                                   help = "batch size in training", type = int, default = 32, required = False)
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
'max_train_tensors', 'max_valid_tensors', 'max_test_tensors', '\\',
'val_fraction', '\\',
'tot_epochs', 'save_each', '\\',
'model_weight', 'optimizer_weight', '\\',
'seed', '\\',
'batch_size', 'learning_rate', 'weight_decay', 'dropout', '\\',
'lr_sch_milestones', 'lr_sch_gamma', '\\']:

    if param_name == '\\':
        print()
    else:
        print(f'{param_name.upper()}: {input_params[param_name]}')

#fix seed for initialization of CNN weights
random.seed(input_params.seed)
np.random.seed(input_params.seed)
torch.manual_seed(input_params.seed)

train_on, valid_on, test_on = 0, 0, 0 #will be set to 1 if the corresponding operation is expected

if input_params.train_dataset:

    train_valid_images = pd.read_csv(input_params.train_dataset, header=None).squeeze(1) #full path to imgb batches

    train_valid_images = train_valid_images.sample(frac=1., random_state=1).tolist() #shuffle

    N_valid = int(input_params.val_fraction*len(train_valid_images)) #number of validation batches

    valid_images, train_images = train_valid_images[:N_valid], train_valid_images[N_valid:]

    print(f'Train imgb batches provided: {len(train_images)}')

    print(f'Validation imgb batches provided: {len(valid_images)}')

    train_on, valid_on = len(train_images)>0, len(valid_images)>0

if input_params.test_dataset:

    test_images = pd.read_csv(input_params.test_dataset, header=None).squeeze(1).tolist() #full path to imgb batches

    print(f'Test/Inference imgb batches provided: {len(test_images)}')

    test_on = len(test_images)>0

assert train_on+valid_on+test_on>0, 'Insufficient number of imgb batches for operation' #not enough tensors for training/evaluation

p_hot_correction_factor = 1e-4 #for p-hot reads encoded as ushort in variant_to_tensor function

class TensorDataset(IterableDataset):

    '''
    Dataset of variant tensors
    '''

    def __init__(self,
                 imgb_list,      #full path to imgb batches
                 max_tensors,    #maximal number of tensors
                ):

        self.imgb_list = imgb_list
        self.target_height = input_params.tensor_height #target tensor height for the CNN
        self.target_width = input_params.tensor_width   #target tensor width for the CNN
        self.max_depth = input_params.max_depth         #read depth normalization constant

        self.max_tensors = max_tensors                  #maximal number of tensors
        self.tensor_counter = 0                         #number of tensors already processed

    def process_data(self, imgb_path):
        '''
        Retrieve tensors consecutively from an imgb batch
        '''
        with open(imgb_path, 'rb') as imgb_header:
            while True:
                if self.max_tensors!=None and self.tensor_counter >= self.max_tensors:
                    break
                try:
                    yield self.get_tensor(imgb_header)
                except EOFError: #pickle file ends
                    break

    def get_tensor(self, imgb_header):

        '''
        Retrieve a single tensor from an imgb batch

        If tensor height is smaller than self.target_height,
        we pad it with 0 to reach the required self.target_height.
        If tensor height is larger that self.target_height,
        we remove some reads on the top and on the bottom, leaving the central part.

        If tensor width is smaller than self.target_width,
        we pad it with 0 to reach the required self.target_width.
        If tensor width is larger that self.target_width,
        we remove some reads on the left and on the right, leaving the central part.

        '''
        tensor, variant_meta = pickle.load(imgb_header) #retrieve tensor and meta information

        one_hot_ref = tensor['one_hot_ref']
        p_hot_reads = tensor['p_hot_reads']
        flags_reads = tensor['flags_reads']

        p_hot_reads = p_hot_reads*p_hot_correction_factor

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
            #THIS CROPPING DOESN'T GUARANTEE THAT VAF IS PRESERVED
            #TO PRESERVE VAF, THE TENSOR HEIGHT SHOULD BE THE SAME AS FOR TENSOR GENERATION
            shift = max(tensor_height//2-self.target_height//2,0)
            full_tensor_h = tensor[shift:shift+self.target_height,:,:]

        if self.target_width>tensor_width:

            #if reads are too short, pad reads with 0 to reach the target width
            padding_tensor = np.zeros((self.target_height, self.target_width-tensor_width, 14))
            full_tensor_w = np.concatenate((full_tensor_h, padding_tensor), axis = 1) #concatenate over the sequence axis
            full_tensor_w = np.roll(full_tensor_w,max(self.target_width//2-tensor_width//2,0),axis=1) #put the piledup reads in the center of tensor

        else:

            #if the reads are too wide, keep the central part, crop on the left and on the right
            shift = max(tensor_width//2-self.target_width//2,0)
            full_tensor_w = full_tensor_h[:,shift:shift+self.target_width,:]

        full_tensor = np.transpose(full_tensor_w, (2,0,1)) #change dimensions order to CxWxH

        label = float('SOMATIC' in variant_meta["info"])

        flanking_data = misc.get_misc_tensor_data(variant_meta, self.max_depth) #extract information added explicitly to fully connected layers (flanking regions)

        self.tensor_counter += 1

        return full_tensor, label, flanking_data, variant_meta

    def __iter__(self):
        return chain.from_iterable(map(self.process_data, self.imgb_list)) #load tensors consecutively from all batches in self.imgb_list

def collate_fn(data):
    '''
    Collate tensors
    '''
    return [torch.tensor(np.array(item, dtype=float), dtype=torch.float) if item_idx<3 else item for item_idx, item in enumerate(zip(*data))] #convert to Tensors all except variant_meta

#define train and evaluation datasets and dataloaders

if train_on:

    train_dataset = TensorDataset(train_images, max_tensors = input_params.max_train_tensors)

    train_dataloader = DataLoader(train_dataset, batch_size=input_params.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

if valid_on:

    valid_dataset = TensorDataset(valid_images, max_tensors = input_params.max_valid_tensors)

    valid_dataloader = DataLoader(valid_dataset, batch_size=512, shuffle=False, num_workers=0, collate_fn=collate_fn)

if test_on:

    test_dataset = TensorDataset(test_images, max_tensors = input_params.max_test_tensors)

    test_dataloader = DataLoader(test_dataset, batch_size=512, shuffle=False, num_workers=0, collate_fn=collate_fn)

#access the GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('\nCUDA device: GPU\n')
else:
    device = torch.device('cpu')
    print('\nCUDA device: CPU\n')

model = models.ConvNN(dropout=input_params.dropout, target_width=input_params.tensor_width, target_height=input_params.tensor_height) #define model architecture

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

        lr_scheduler.step() #for MultiStepLR we make a step after each epoch

        train_ROC_AUC, _ = misc.get_ROC(train_pred)

        print(f'EPOCH: {epoch} - train loss: {train_loss:.4}, train ROC AUC: {train_ROC_AUC:.4}')

        if input_params.save_each!=0 and (epoch%input_params.save_each==0 or epoch==tot_epochs): #save model weights

            misc.save_model_weights(model, optimizer, weights_dir, epoch)

            misc.save_predictions(train_pred, predictions_dir, f'training_epoch_{epoch}.vcf') #save train predictions

    if valid_on:

        print(f'EPOCH {epoch}: Validating...')

        valid_loss, valid_pred = train_eval.model_eval(model, optimizer, valid_dataloader, device)

        valid_ROC_AUC, _ = misc.get_ROC(valid_pred)

        print(f'EPOCH: {epoch} - validation loss: {valid_loss:.4}, validation ROC AUC: {valid_ROC_AUC:.4}')

        misc.save_predictions(valid_pred, predictions_dir, f'validation_epoch_{epoch}.vcf') #save validation predictions

    if test_on and epoch==tot_epochs:

        print(f'EPOCH {epoch}: Test/Inference...')

        start_time = time.time()

        test_loss, test_pred = train_eval.model_eval(model, optimizer, test_dataloader, device)

        tot_test_time = time.time() - start_time

        test_ROC_AUC, _ = misc.get_ROC(test_pred)

        if test_ROC_AUC!=-1:

            print(f'EPOCH: {epoch} - test loss: {test_loss:.4}, test ROC AUC: {test_ROC_AUC:.4}')

        misc.save_predictions(test_pred, predictions_dir, 'final_predictions.vcf') #save test/inference predictions


print(f'peak GPU memory allocation: {round(torch.cuda.max_memory_allocated(device)/1024/1024)} Mb')
print(f'total train time: {round(tot_train_time)} s : {round(len(train_pred)*(tot_epochs-last_epoch)/(tot_train_time+1))} samples/s')
print(f'total inference time: {round(tot_test_time)} s : {round(len(test_pred)/(tot_test_time+1))} samples/s')
print('Done')
