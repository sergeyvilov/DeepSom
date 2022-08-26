import os, sys
import pandas as pd
import numpy as np
import re

import builtins
import time

import torch
from sklearn.metrics import roc_curve, auc

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

class dotdict(dict):
    '''
    Dictionary with dot.notation access to attributes
    '''

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def print(*args, **kwargs):
    '''
    Redefine print function for logging
    '''
    now = time.strftime("[%Y/%m/%d-%H:%M:%S]-", time.localtime()) #place current time at the beggining of each printed line
    builtins.print(now, *args, **kwargs)
    sys.stdout.flush()

def extract_flanking_info(info):
    '''
    Extract information about flanking regions from an INFO string
    '''
    #if 'flanking=' in info:
    info = re.search('flanking=([0-9\.\-]*)\|([0-9\.\-]*)\|([0-9\.\-]*)\|([0-9\.\-]*)',info)
    #else:
    #    info = re.search('flanking_lVAF=([0-9\.\-]*);flanking_lDP=([0-9\.\-]*);flanking_rVAF=([0-9\.\-]*);flanking_rDP=([0-9\.\-]*)',info)
    if info==None:
        return (-1,-1,-1,-1) #negative value when the data is missing
    return info.groups()

def normalize_dp(dp, max_depth):
    if dp>0:
        return min(dp/max_depth,1) #normalize clip the right tail
    else:
        #negative value when the data is missing
        return max(dp,-1) #clip the left tail

def get_misc_tensor_data(imgb_batch_meta, max_depth):
    '''
    Extract information about flanking regions for all variants in the imgb batch
    '''
    info = [(d['VAF0'],d['DP0'],*extract_flanking_info(d['info'])) for d in imgb_batch_meta]

    info = [list(map(float,item)) for item in info]

    if max_depth < 0:
        info = [[vaf, normalize_dp(dp, abs(max_depth)), -1, -1, -1, -1] for vaf,dp,lvaf,ldp,rvaf,rdp in info]
        #info = [[-1, -1, -1, -1, -1, -1] for vaf,dp,lvaf,ldp,rvaf,rdp in info]
    else:
        info = [[vaf, normalize_dp(dp, max_depth), lvaf, normalize_dp(ldp, 2*max_depth), rvaf, normalize_dp(rdp, 2*max_depth)] for vaf,dp,lvaf,ldp,rvaf,rdp in info]

    return info

# def resample(df,                #dataframe with 'labels' column
#             resample_mode       #None, 'upsample' or 'downsample'
#             ):
#     """
#     Equilibrate classes in the dataframe by resampling
#
#     resample_mode:
#
#     None: do not resample
#     "upsample": equilibrate classes by upsampling to the overrepresented class
#     "downsample": equilibrate classes by downsampling to the underrepresented class
#
#     """
#
#     if len(df)==0 or str(resample_mode)=='None':
#         return df
#
#     current_class_counts = df['label'].value_counts()
#
#     if resample_mode == 'upsample':
#
#         new_class_counts = [(class_name, current_class_counts.max()) for class_name in
#                                df['label'].unique()]
#
#     elif resample_mode == 'downsample':
#
#         new_class_counts = [(class_name, current_class_counts.min()) for class_name in
#                                df['label'].unique()]
#
#     else:
#
#         raise Exception(f'Resample mode not recognized: {resample_mode}')
#
#     resampled_df = pd.DataFrame()
#
#     for class_name, class_counts in new_class_counts:
#
#         class_df = df.loc[df['label']==class_name]
#
#         replace = class_counts>current_class_counts[class_name] #will be True only for upsampling
#
#         resampled_class_df = class_df.sample(n=class_counts, replace=replace, random_state=1)
#
#         resampled_df = pd.concat([resampled_df, resampled_class_df])
#
#     resampled_df = resampled_df.sample(frac=1, random_state=1)
#
#     return resampled_df


def get_ROC(predictions):

    '''
    Compute ROC from NN predictions,

    return AUC and interpolated ROC curve
    '''

    if len(predictions)==0:
        return -1.0, 'ROC curve can not be displayed'

    _, y_pred, y_true = zip(*predictions)

    y_pred, y_true = np.array(y_pred), np.array(y_true)

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    auROC = auc(fpr, tpr)

    tpr_new = np.unique(np.hstack((np.linspace(0.01,0.05,5), np.linspace(0.05,0.95,19),  np.linspace(0.95,0.99,5)))) #interpolation grid

    thr_new = np.interp(tpr_new, tpr[1:], thresholds[1:] )

    tp = np.array([((y_true==(y_pred>thr))&(y_true==1)).sum() for thr in thr_new]) #true positives
    fp = np.array([((y_true!=(y_pred>thr))&(y_true==0)).sum() for thr in thr_new]) #false positives

    tpr_new = tp/y_true.sum()
    fpr_new = fp/(y_true==0).sum()

    ROC_df = pd.DataFrame({'output_threshold':thr_new, 'true_positives':tp, 'false_positives':fp, 'true_positive_rate':tpr_new, 'false_positive_rate':fpr_new})

    return auROC, ROC_df


def save_predictions(predictions, dataset, output_dir, output_name):

    '''
    Save predictions in a vcf file
    '''

    output_list = list()

    predictions_snps, predictions_indels = list(), list()

    #loop over all predictions
    for tensor_pos, score, label in predictions:
        variant_meta = dataset.variant_meta[tensor_pos[0]][tensor_pos[1]] #locate variant meta information
        variant_row = [variant_meta['chrom'], variant_meta['pos'], '.', variant_meta['ref'], variant_meta['alt'], '.', '.']  #this will be a row in the output vcf file
        if len(variant_meta['ref']) == len(variant_meta['alt']):
            predictions_snps.append((None,score,label))
        else:
            predictions_indels.append((None,score,label))
        if 'info' in variant_meta.keys():
            variant_info = variant_meta['info'] + ';'
        else:
            variant_info = ''
        for key in ['vcf', 'BAM', 'DP', 'VAF', 'DP0', 'VAF0', 'refseq', 'batch_name', 'imgb_index', 'GERMLINE', 'Sample']:
            #all supplementary information goes to the INFO field
            if key in variant_meta.keys():
                variant_info += f"{key}={variant_meta[key]};"
        variant_info += f'nnc_score={score:.4}'
        if label!=None:
            variant_info += f';true_label={int(label)}'
        variant_row.append(variant_info)
        output_list.append(variant_row)

    roc_auc_snps, roc_curve_snps = get_ROC(predictions_snps)

    roc_auc_indels, roc_curve_indels = get_ROC(predictions_indels)

    print(f'ROC AUC SNPs: {roc_auc_snps:.4}\n')
    builtins.print(roc_curve_snps)

    print(f'ROC AUC INDELs: {roc_auc_indels:.4}\n')
    builtins.print(roc_curve_indels)

    output_df = pd.DataFrame(output_list, columns=['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO']) #list to a DataFrame

    chrom_dict = {'X':23,'Y':24,'M':25,'MT':25,'chrX':23,'chrY':24,'chrM':25,'chrMT':25}
    output_df.sort_values(by=['#CHROM', 'POS'], key=lambda a:a.apply(lambda x:int(x) if type(x)==int or x.isnumeric() else chrom_dict.get(x,100)), inplace=True) #sort variants by chrom

    output_name = os.path.join(output_dir, output_name)

    #first write vcf header
    with open(output_name, 'w') as f:
            f.write('##fileformat=VCFv4.2\n')
            f.write('##INFO=<ID=vcf,Number=.,Type=String,Description="Name of the vcf file from which the variant comes">\n')
            f.write('##INFO=<ID=BAM,Number=.,Type=String,Description="BAM file name for the variant">\n')
            f.write('##INFO=<ID=Sample,Number=.,Type=String,Description="Sample name for the variant">\n')
            f.write('##INFO=<ID=Project,Number=.,Type=String,Description="Project name for the variant">\n')
            f.write('##INFO=<ID=GERMLINE,Number=1,Type=Integer,Description="Germline variant">\n')
            f.write('##INFO=<ID=SOMATIC,Number=1,Type=Integer,Description="Somatic variant">\n')
            f.write('##INFO=<ID=DP0,Number=1,Type=Integer,Description="DP from BAM file">\n')
            f.write('##INFO=<ID=VAF0,Number=1,Type=Float,Description="VAF from BAM file">\n')
            f.write('##INFO=<ID=gnomAD_AF,Number=1,Type=Float,Description="Alternative allele frequency as in GNOMAD">\n')
            f.write('##INFO=<ID=refseq,Number=.,Type=String,Description="Reference sequence around the variant site, 30 bases to the left and $half_n_ref_bases bases to the right">\n')
            f.write('##INFO=<ID=batch_name,Number=.,Type=String,Description="Name of the imgb batch containing the variant">\n')
            f.write('##INFO=<ID=imgb_index,Number=1,Type=Integer,Description="Index of the variant in the imgb batch">\n')
            f.write('##INFO=<ID=nnc_score,Number=1,Type=Float,Description="Neural Network classification score">\n')
            f.write('##INFO=<ID=true_label,Number=1,Type=Integer,Description="True label (0 or 1)">\n')

    output_df.to_csv(output_name, mode='a', sep='\t', index=False) #append predictions to the vcf file

    print(f'Predictions saved in {output_name}')

def save_model_weights(model, optimizer, output_dir, epoch):

    '''
    Save model and optimizer weights
    '''

    config_save_base = os.path.join(output_dir, f'epoch_{epoch}_weights')

    print(f'epoch:{epoch}: SAVING MODEL, CONFIG_BASE: {config_save_base}\n')

    torch.save(model.state_dict(), config_save_base+'_model') #save model weights

    torch.save(optimizer.state_dict(), config_save_base+'_optimizer') #save optimizer weights
