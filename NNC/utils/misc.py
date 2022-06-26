import os
import pandas as pd

import torch
from sklearn.metrics import roc_curve, auc

def resample(df,                #dataframe with 'labels' column
            resample_mode       #None, 'upsample' or 'downsample'
            ):
    """
    Equilibrate classes in the dataframe by resampling

    resample_mode:

    None: do not resample
    "upsample": equilibrate classes by upsampling to the overrepresented class
    "downsample": equilibrate classes by downsampling to the underrepresented class

    """

    if len(df)==0 or str(resample_mode)=='None':
        return df

    current_class_counts = df['label'].value_counts()

    if resample_mode == 'upsample':

        new_class_counts = [(class_name, current_class_counts.max()) for class_name in
                               df['label'].unique()]

    elif resample_mode == 'downsample':

        new_class_counts = [(class_name, current_class_counts.min()) for class_name in
                               df['label'].unique()]

    else:

        raise Exception(f'Resample mode not recognized: {resample_mode}')

    resampled_df = pd.DataFrame()

    for class_name, class_counts in new_class_counts:

        class_df = df.loc[df['label']==class_name]

        replace = class_counts>current_class_counts[class_name] #will be True only for upsampling

        resampled_class_df = class_df.sample(n=class_counts, replace=replace, random_state=1)

        resampled_df = pd.concat([resampled_df, resampled_class_df])

    resampled_df = resampled_df.sample(frac=1, random_state=1)

    return resampled_df


def get_ROC(predictions):

    '''
    Compute ROC AUC from NN predictions
    '''

    _, scores, labels = zip(*predictions)

    fpr, tpr, _ = roc_curve(labels, scores)

    auROC = auc(fpr, tpr)

    return auROC


def save_predictions(predictions, dataset, output_dir, epoch, inference_mode=False):

    '''
    Save predictions in a vcf file
    '''

    #imgb_names,_=zip(*dataset.data)

    #predictions = [(imgb_names[tensor_pos[0]], tensor_pos[1], score, label) for tensor_pos, score, label in predictions] #(imgb_name, pos_in_imgb, nn_score, true_label)

    output_list = list()

    #loop over all predictions
    for tensor_pos, score, label in predictions:
        variant_meta = dataset.variant_meta[tensor_pos[0]][tensor_pos[1]] #locate variant meta information
        variant_row = [variant_meta['chrom'], variant_meta['pos'], '.', variant_meta['ref'], variant_meta['alt'], '.', '.']  #this will be a row in the output vcf file
        variant_info = '' #all supplementary information goes to the INFO field
        for key in ['vcf', 'BAM', 'DP', 'VAF', 'batch_name', 'imgb_index', 'GERMLINE', 'Sample']:
            if key in variant_meta.keys():
                variant_info += f"{key}={variant_meta[key]};"
        variant_info += f'nnc_score={score:.4}'
        if label:
            variant_info += ';true_label={label}'
        variant_row.append(variant_info)
        output_list.append(variant_row)

    output_df = pd.DataFrame(output_list, columns=['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO']) #list to a DataFrame

    chrom_dict = {'X':23,'Y':24,'M':25,'MT':25,'chrX':23,'chrY':24,'chrM':25,'chrMT':25}
    output_df.sort_values(by=['#CHROM', 'POS'], key=lambda a:a.apply(lambda x:int(x) if type(x)==int or x.isnumeric() else chrom_dict.get(x,100)), inplace=True) #sort variants by chrom

    if not inference_mode:
        #if output_name not provided, infer output_name from output_dir and epoch
        output_name = os.path.join(output_dir, f'epoch_{epoch}.vcf')
    else:
        #make sure that the path to the predictions vcf exists
        #os.makedirs(os.path.dirname(output_name), exist_ok=True)
        output_name = os.path.join(output_dir, f'inference.vcf')

    #write vcf header
    with open(output_name, 'w') as f:
            f.write('##fileformat=VCFv4.2\n')
            f.write('##INFO=<ID=vcf,Number=.,Type=String,Description="Name of the vcf file from which the variant comes">\n')
            f.write('##INFO=<ID=BAM,Number=.,Type=String,Description="BAM file name for the variant">\n')
            f.write('##INFO=<ID=DP,Number=1,Type=Integer,Description="Approximate read depth; some reads may have been filtered">\n')
            f.write('##INFO=<ID=VAF,Number=1,Type=Float,Description="VAF from mutect read filter if available">\n')
            f.write('##INFO=<ID=gnomAD_AF,Number=1,Type=Float,Description="Alternative allele frequency as in GNOMAD">\n')
            f.write('##INFO=<ID=batch_name,Number=.,Type=String,Description="Name of the imgb batch containing the variant">\n')
            f.write('##INFO=<ID=imgb_index,Number=1,Type=Integer,Description="Index of the variant in the imgb batch">\n')
            f.write('##INFO=<ID=nnc_score,Number=1,Type=Float,Description="Neural Network classification score">\n')
            f.write('##INFO=<ID=true_label,Number=1,Type=Integer,Description="True label (0 or 1)">\n')

    output_df.to_csv(output_name, mode='a', sep='\t', index=False) #append predictions to the vcf file

    #pd.DataFrame(predictions, columns=['imgb_path', 'imgb_index', 'score', 'label']).to_csv(output_name, index=False)


def save_model_weights(model, optimizer, output_dir, epoch):

    '''
    Save model and optimizer weights
    '''

    config_save_base = os.path.join(output_dir, f'epoch_{epoch}_weights')

    print(f'epoch:{epoch}: SAVING MODEL, CONFIG_BASE: {config_save_base}\n')

    torch.save(model.state_dict(), config_save_base+'_model') #save model weights

    torch.save(optimizer.state_dict(), config_save_base+'_optimizer') #save optimizer weights
