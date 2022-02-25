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
    Replace imgb index with imgb path in NN predictions and save predictions as a csv file
    '''

    imgb_names,_=zip(*dataset.data)

    predictions = [(imgb_names[tensor_pos[0]], tensor_pos[1], score, label) for tensor_pos, score, label in predictions] #(imgb_name, pos_in_imgb, nn_score, true_label)

    if not inference_mode:
        #if output_name not provided, infer output_name from output_dir and epoch
        output_name = os.path.join(output_dir, f'epoch_{epoch}.csv')
    else:
        #make sure that the path to the predictions csv exists
        #os.makedirs(os.path.dirname(output_name), exist_ok=True)
        output_name = os.path.join(output_dir, f'inference.csv')

    pd.DataFrame(predictions, columns=['imgb_path', 'imgb_index', 'score', 'label']).to_csv(output_name, index=False)


def save_model_weights(model, optimizer, output_dir, epoch):

    '''
    Save model and optimizer weights
    '''

    config_save_base = os.path.join(output_dir, f'epoch_{epoch}_weights')

    print(f'epoch:{epoch}: SAVING MODEL, CONFIG_BASE: {config_save_base}\n')

    torch.save(model.state_dict(), config_save_base+'_model') #save model weights

    torch.save(optimizer.state_dict(), config_save_base+'_optimizer') #save optimizer weights
