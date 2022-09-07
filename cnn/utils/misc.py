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
    now = time.strftime("[%Y/%m/%d-%H:%M:%S]-", time.localtime()) #current date and time at the beggining of each printed line
    builtins.print(now, *args, **kwargs)
    sys.stdout.flush()


def get_misc_tensor_data(variant_meta, max_depth):
    '''
    Extract information about flanking regions from an INFO string
    '''

    VAF0, DP0 = float(variant_meta['VAF0']), float(variant_meta['DP0'])

    DP0 = min(DP0 / max_depth, 1)

    flanking = re.search('flanking=([0-9|-]+)',variant_meta['info'])

    if max_depth>0 and flanking!=None:

        flanking = np.array([float(x) for x in flanking[1].split('|')])

        AD_alt = flanking[1::2] #odd position - alternative allele depth
        AD_ref = flanking[::2] #even position - reference allele depth

        DP = AD_ref + AD_alt

        VAF = AD_alt / DP

        mask = (AD_ref>=0)&(AD_alt>=0)

        DP = np.where(mask, np.clip(DP / max_depth, 0, 1), -1)

        VAF = np.where(mask, VAF, -1) #-1 when AD_ref=AD_alt=-1

        return np.hstack((DP0, DP, VAF0, VAF)).tolist()

    else:

        return [DP0, -1,-1,-1,-1, VAF0, -1,-1,-1,-1]

def get_ROC(predictions):

    '''
    Compute ROC from CNN predictions

    return AUC and interpolated ROC curve
    '''

    if len(predictions)==0:
        return -1., 'ROC can not be computed: no predictions were made'

    y_pred, y_true, _ = zip(*predictions)

    if all(y_true) or not any(y_true):
        return -1., 'ROC can not be computed: variant labels are all equal or absent (possibly inference dataset)'

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


def save_predictions(predictions, output_dir, output_name):

    '''
    Save predictions in a vcf file
    '''

    output_list = list()

    predictions_snps, predictions_indels = list(), list()

    #loop over all predictions
    for score, label, variant_meta in predictions:
        variant_row = [variant_meta['chrom'], variant_meta['pos'], '.', variant_meta['ref'], variant_meta['alt'], '.', '.']  #this will be a row in the output vcf file
        if len(variant_meta['ref']) == len(variant_meta['alt']):
            predictions_snps.append((score,label,None))
        else:
            predictions_indels.append((score,label,None))
        if 'info' in variant_meta.keys():
            variant_info = variant_meta['info'] + ';'
        else:
            variant_info = ''
        for key in ['vcf', 'DP0', 'VAF0', 'batch_name', 'imgb_index']:
            #all supplementary information from variant_meta goes to the INFO field
            if key in variant_meta.keys():
                variant_info += f"{key}={variant_meta[key]};"
        variant_info += f'cnn_score={score:.4}'

        variant_row.append(variant_info)
        output_list.append(variant_row)

    roc_auc_snps, roc_curve_snps = get_ROC(predictions_snps)

    roc_auc_indels, roc_curve_indels = get_ROC(predictions_indels)

    print(f'ROC AUC SNPs: {roc_auc_snps:.4}\n')
    builtins.print(roc_curve_snps)

    print(f'ROC AUC INDELs: {roc_auc_indels:.4}\n')
    builtins.print(roc_curve_indels)

    output_df = pd.DataFrame(output_list, columns=['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO']) #list to DataFrame

    chrom_dict = {'X':23,'Y':24,'M':25,'MT':25,'chrX':23,'chrY':24,'chrM':25,'chrMT':25}
    output_df.sort_values(by=['#CHROM', 'POS'], key=lambda a:a.apply(lambda x:int(x) if type(x)==int or x.isnumeric() else chrom_dict.get(x,100)), inplace=True) #sort variants by chrom

    output_name = os.path.join(output_dir, output_name)

    #first write vcf header
    with open(output_name, 'w') as f:
            f.write('##fileformat=VCFv4.2\n')
            f.write('##INFO=<ID=vcf,Number=.,Type=String,Description="Name of the vcf file from which the variant comes">\n')
            f.write('##INFO=<ID=BAM,Number=.,Type=String,Description="BAM file name">\n')
            f.write('##INFO=<ID=flanking,Number=.,Type=String,Description="Ref and alt AD of left and right flanking variants">\n')
            f.write('##INFO=<ID=GERMLINE,Number=.,Type=Flag,Description="Germline variant">\n')
            f.write('##INFO=<ID=SOMATIC,Number=.,Type=Flag,Description="Somatic variant">\n')
            f.write('##INFO=<ID=DP0,Number=1,Type=Integer,Description="DP based on BAM">\n')
            f.write('##INFO=<ID=VAF0,Number=1,Type=Float,Description="VAF based on BAM">\n')
            f.write('##INFO=<ID=gnomAD_AF,Number=1,Type=Float,Description="gnomAD population allele frequency">\n')
            #f.write('##INFO=<ID=refseq,Number=.,Type=String,Description="Reference sequence around the variant">\n')
            f.write('##INFO=<ID=batch_name,Number=.,Type=String,Description="Name of the imgb batch containing the variant">\n')
            f.write('##INFO=<ID=imgb_index,Number=1,Type=Integer,Description="Index of the variant in the imgb batch">\n')
            f.write('##INFO=<ID=cnn_score,Number=1,Type=Float,Description="CNN classification score">\n')

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
