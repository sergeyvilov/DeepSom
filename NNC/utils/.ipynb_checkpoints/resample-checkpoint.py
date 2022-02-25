import pandas as pd

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
        
    if len(df)==0 or resample_mode==None:
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