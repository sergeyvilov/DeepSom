import numpy as np

import matplotlib.pyplot as plt

#Helper function for reads encoding

def encode_bases(letters):
    '''
    Encode bases with digits
    '''
    nmap={'A':0,'C':1,'T':2,'G':3, 'N':4, 'K':5, '*':6, 'M':7} #asterix for deletion
    if len(letters)>1:
        return([nmap[n] for n in letters])
    else:      
        return nmap[letters] #ACTGNK*M-->01234567
    
def decode_bases(numbers):
    '''
    Decode bases from digits
    '''
    unmap=('A','C','T','G','N','K','*', 'M')
    if isinstance(numbers, (list, tuple, np.ndarray)):
        return([unmap[int(n)] for n in numbers])
    else:      
        return unmap[int(numbers)] #01234567-->ACTGNK*M

#Helper functions for visualization

def show_diff_image(reads_im, ref_bases):
    '''
    Visualize difference between reference bases and actual reads as a 3-color image
    '''
    diff=reads_im[:,:,0]-ref_bases
    diff[diff!=0]=0.5 #where bases are different btw reads and reference
    diff[diff==0]=1
    diff[reads_im[:,:,0]==4]=0 #where there is no data (N)

    fig = plt.figure(figsize = (10,10))
    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(diff, cmap='viridis')
    

def show_diff_text(reads_im, ref_bases,
                  N_crop = 50, #bases to omit on each side of the variant position, to make lines shorter
                  highlight_column = None #choose any column to highlight with light blue
                  ):
    '''
    Visualize difference between reference bases and actual reads as text
    '''
    diff = reads_im[:,:,0]-ref_bases
    diff = (diff!=0)
    variant_column = reads_im.shape[1]//2

    ref_letters = decode_bases(ref_bases)  #map reads digits back to letters
    ref_letters[variant_column] = "\x1b[34m{}\x1b[0m".format(ref_letters[variant_column]) #dye the site at variant position in blue using ANSI escape code
    ref_str = ''.join(ref_letters) #ref bases to a string

    print(ref_str[N_crop:-N_crop]) #print reference bases


    for idx_read in range(0,reads_im.shape[0],1):
        read = reads_im[idx_read,:,0] #get the read sequence
        read = decode_bases(read) #map reference digits back to letters
        read = read[N_crop:-N_crop] 
        for c_ind in range(len(read)):
            #read[c_ind] = (read[c_ind] if N_crop+c_ind!=highlight_column else "\x1b[36m{}\x1b[0m".format(read[c_ind]))
            #highligh where the read base is different from reference
            read[c_ind] = (read[c_ind] if not diff[idx_read,N_crop+c_ind] or read[c_ind] in ('N') else "\x1b[31m{}\x1b[0m".format(read[c_ind]))#dye the site at mismatches in red using ANSI escape code
        read_str = ''.join(read)
        print(read_str)

        
def pcolor(image, yticklabels=[],xticklabels=[], cmap='binary',figsize = (10,10)):
    '''
    Visualize image on a checked grid, similar to MATLAB pcolor funtion
    '''
    
    fig = plt.figure(figsize = figsize)
    
    ax = plt.gca()
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.imshow(image, cmap = cmap)
    
    #set major ticks to visualize the cell edges correctly
    ax.set_yticks(np.arange(-0.5, image.shape[0]-1, 1), minor=False)
    ax.set_xticks(np.arange(-0.5, image.shape[1]-1, 1), minor=False)
    
    ax.set_yticklabels([], minor=False)
    ax.set_xticklabels([], minor=False)
    
    ax.tick_params(axis="y",direction="in", pad=-22)
    ax.tick_params(axis="x",direction="in", pad=-22)
    
    #minor ticks are real ticks with text
    if yticklabels:
        ax.set_yticks(np.arange(-0.1, image.shape[0]-1, 1), minor=True)
        ax.set_yticklabels(yticklabels, minor=True)
    if xticklabels:
        ax.set_xticks(np.arange(-0.1, image.shape[1]-1, 1), minor=True)
        ax.set_xticklabels(xticklabels, minor=True)
    
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    