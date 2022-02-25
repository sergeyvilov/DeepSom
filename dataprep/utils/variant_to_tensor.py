#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pysam
import re
import os

def is_valid_snp(ref, alt):
    '''
    Check if a snp variant is correct

    invalid snp:
    -REF or ALT allele is not in A,C,T or G
    -REF or ALT allele is longer than 1
    '''
    #alt = alts[0] if isinstance(alts,(tuple,np.ndarray)) else alts

    assert isinstance(ref,str), 'Ref allele is not a string'
    assert isinstance(alt,str), 'Alt allele is not a string'

    if len(ref)!=1:
        raise Exception('More than 1 base in ref allele')

    if len(alt)!=1:
        raise Exception('More than 1 base in alt allele')

    if not ref in ['A', 'C', 'T', 'G']:
        raise Exception('Wrong base in ref allele')

    if not alt in ['A', 'C', 'T', 'G']:
        raise Exception('Wrong base in alt allele')

    return 0

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

def get_phred_qual(qual_symbol):
    '''
    Convert an ASCII quality character to the probability that a base is called incorrectly
    '''
    probability = 10 ** ((ord(qual_symbol) - 33) / -10)

    return probability

def compute_VAF_DP(pileup_column, alt):
    '''
    Return VAF and DP for a pileup column
    '''
    DP = len(pileup_column)
    VAF = (pileup_column == encode_bases(alt)).sum()/DP

    return VAF,DP

def variant_to_tensor(variant, ref_fasta_file, bam_file,
                            tensor_width = 150, # tensor width: 2x the most probable read length
                            tensor_max_height = 30, #max tensor height, the probability to have a read depth above this value should be small
                            tensor_sort_by_variant = False, #sort reads by value in the variant column
                            tensor_check_variant_column = False, #check if there's any alternative allele in the variant column (some variants may be corrupted due to a bad variant position etc)
                            tensor_crop_strategy = 'topbottom', #crop strategy when read depth exceeds tensor_max_height
                            check_variant = None, #check if the variant is correct,possible values: "snps", "indels", None
                            ):

    '''
    Collect reads for a given variant and transform them to a variant image

    When reading the BAM file, consider only first MAX_RAW_READS reads.
    Discard reads with at least one flag from EXCLUDE_FLAGS.

    Use p-hot encoding for read bases and one-hot encoding for reference bases.

    Crop tensors whose height is above tensor_max_height using tensor_crop_strategy.

    Return tensor, reference sequence, VAF and DP before cropping.

    To learn more about encoding, see
    Friedman S. et al. Lean and deep models for more accurate filtering of SNP
    and INDEL variant calls //Bioinformatics. – 2020. – Т. 36. – №. 7. – С. 2060-2067.
    '''

    MAX_RAW_READS=1000 #maximum number of raw reads to consider for a variant

    EXCLUDE_FLAGS = 0x4|0x200|0x400 #exclude reads with flags "segment unmapped", "not passing filters", "PCR or optical duplicate"

    #basic check and filtering of invalid variants
    if check_variant=="snps":
        is_valid_snp(variant['ref'], variant['alt'])

    if not os.path.isfile(bam_file):
        raise Exception('BAM file not found')

    variant_column_idx = tensor_width//2 #variant column in the middle of the tensor

    def get_ref_bases(variant):

        '''
        Get reference sequence around the variant positon, variant being roughly in the center
        '''
        reffile = pysam.FastaFile(ref_fasta_file) #open reference genome FASTA

        ref_bases = reffile.fetch(variant['chrom'], variant['refpos']-variant_column_idx-1, variant['refpos']-variant_column_idx+tensor_width-1) #reference bases around the variant position

        ref_bases = ref_bases.upper() #ignore strand information

        if ref_bases[variant_column_idx]!=variant['ref']:
            raise Exception('Variant reference allele not found in the right position in the reference bases string!')

        #print('Reference sequence: ' + ''.join(ref_bases))

        ref_bases = np.array(encode_bases(ref_bases)) # letters to digits

        return ref_bases

    ref_bases = get_ref_bases(variant)

    samfile = pysam.AlignmentFile(bam_file, "rb" ) #open the variant BAM file

    raw_reads = []

    #collect all the reads around the candidate variant position
    for read in samfile.fetch(variant['chrom'], variant['pos']-2, variant['pos']+2):
        if (read.pos<=variant['pos']-1 and read.reference_end>variant['pos'] and
            read.flag&EXCLUDE_FLAGS==0):
                raw_reads.append((read.pos,read.seq,read.qual,read.flag,read.cigartuples))
        if len(raw_reads)>=MAX_RAW_READS:#sometimes there are a lot of reads, we don't need all of them
            break


    #Align reads according to their CIGAR strings.
    #For each read its cigar string is analysed to place
    #read bases correctly (taking into account clips, insertions and deletions).

    aligned_reads = []

    for read_idx, read in enumerate(raw_reads):

        pos,seq,qual,flag,cigartuples = read

        aligned_seq = []  #aligned read sequence
        aligned_qual = [] #aligned read qualities

        seq = seq.upper()
        seq = encode_bases(seq) #letters to digits

        qual = np.array([get_phred_qual(q) for q in qual])     #probability that the base is called INCORRECTLY
        qual = 1.-qual                                         #probability that the base is called CORRECTLY

        c = 0 #current position in the original (not aligned) read

        #we move along the original read sequence and make insertions/delections when necessary
        for op in cigartuples:
            optype, oplen = op #type and length of cigar operation
            if optype==5:#hard clip: do nothing as it's not included in seq
                continue
            elif optype==4:#soft clip: exclude these positions from aligned seq as they aren't used by callers
                c+=oplen
            elif optype==2 or optype==3 or optype==6: #deletion or padding
                aligned_seq.extend([encode_bases('*')]*oplen)
                aligned_qual.extend([0]*oplen)
            elif optype==1:#insertion
                c+=oplen
            else: #match or mismatch
                aligned_seq.extend(seq[c:c+oplen])
                aligned_qual.extend(qual[c:c+oplen])
                c+=oplen

        aligned_reads.append((pos,aligned_seq,aligned_qual,flag))

    N_reads = len(aligned_reads)

    if not N_reads:
        raise Exception('No reads for this variant')

    variant_column = np.array([read_seq[(variant['pos']-1)-read_pos] for read_pos, read_seq, *_ in aligned_reads])

    VAF0,DP0 = compute_VAF_DP(variant_column, variant['alt']) #VAF and DP before cropping

    #print(f'Before cropping: Read depth (DP): {DP0}, VAF:{VAF0}')

    #if there are more reads than we can include in the tensor, we have to remove some of them

    if tensor_crop_strategy == 'center':
        #keep reads at the top and at the bottom, remove in the center
        aligned_reads = aligned_reads[:tensor_max_height//2] + aligned_reads[max(N_reads_tot-tensor_max_height//2,0):N_reads]
    elif tensor_crop_strategy == 'topbottom':
        #keep reads in the center, remove at the top and at the bottom
        shift = max(N_reads//2-tensor_max_height//2,0)
        aligned_reads = aligned_reads[shift:shift+tensor_max_height]

    variant_column = np.array([read_seq[(variant['pos']-1)-read_pos] for read_pos, read_seq, *_ in aligned_reads])

    VAF,DP = compute_VAF_DP(variant_column, variant['alt']) #VAF and DP after cropping

    #print(f'After cropping: Read depth (DP): {DP}, VAF:{VAF}')

    N_reads = len(aligned_reads)

    if tensor_check_variant_column:
        #check if (after cropping) we still have the alternative allele in the variant column
        variant_column = np.array([read_seq[(variant['pos']-1)-read_pos] for read_pos, read_seq, *_ in aligned_reads])
        is_alt = (variant_column==encode_bases(variant['alt']))
        if (isinstance(is_alt, bool) and not is_alt) or not is_alt.any():
            raise Exception('No reads with the alternative allele found in the variant column!')

    if tensor_sort_by_variant:
        #sort tensor by base in the variant column
        variant_column = np.array([read_seq[(variant['pos']-1)-read_pos] for read_pos, read_seq, *_ in aligned_reads])
        diff = (variant_column==encode_bases(variant['ref'])).astype(int)-(variant_column>3).astype(int).tolist() #all N,M,K letters will go to the bottom
        aligned_reads_sorted_tuple = sorted(zip(aligned_reads,diff),key=lambda x:x[1], reverse=True)
        aligned_reads = [read[0] for read in aligned_reads_sorted_tuple]

    reads_im = np.zeros((N_reads,tensor_width,2)) #2 channels to encode the sequence and probability of each read base

    reads_im[:,:,0] = encode_bases('N') #bases for all reads, default: no data ('N')
    reads_im[:,:,1] = 1/4. #probability to have the corresponding basis, default:equal probability for all bases

    #pileup reads
    for read_idx, read in enumerate(aligned_reads):

        pos,seq,qual,flag = read #absolute position, sequence, quality scores and flags of the read

        start_pos = pos-(variant['pos']-1-variant_column_idx) #relative position of the read in the tensor

        if start_pos<0: #left end of the read is beyond the defined tensor edge

            #reject data that's beyond the tensor edge
            seq = seq[-start_pos:]
            qual = qual[-start_pos:]
            start_pos = 0

        rlen = len(seq)

        reads_im[read_idx,start_pos:start_pos+rlen,0] = seq[:tensor_width-start_pos]
        reads_im[read_idx,start_pos:start_pos+rlen,1] = qual[:tensor_width-start_pos]

    p_hot_reads = np.zeros((N_reads,tensor_width,4)) # p-hot encoding of read bases probabilities: each channel gives the probability of the corresponding base (ACTG)

    for basis_idx in range(4):
        p_hot_reads[:,:,basis_idx] = np.where(reads_im[:,:,0]==basis_idx, reads_im[:,:,1], (1.-reads_im[:,:,1])/3.)

    del_row, del_col = np.where(reads_im[:,0,:]==6) #deletions
    p_hot_reads[del_row, del_col, :] = 0.

    M_row, M_col = np.where(reads_im[:,0,:]==7) #M: either A or C, each with probability 0.5
    p_hot_reads[M_row, M_col, 0] = 1.
    p_hot_reads[M_row, M_col, 1] = 1.

    K_row, K_col = np.where(reads_im[:,0,:]==5) #K: either G or T, each with probability 0.5
    p_hot_reads[K_row, K_col, 2] = 1.
    p_hot_reads[K_row, K_col, 3] = 1.

    one_hot_ref = np.zeros((1,tensor_width,4)) #one-hot encoding of reference bases, same for all reads

    for basis_idx in range(4):
        one_hot_ref[:,(ref_bases==basis_idx),basis_idx] = 1.

    one_hot_ref[:,(ref_bases==4),:] = 1 #N
    one_hot_ref[:,(ref_bases==7),0] = 1 #M
    one_hot_ref[:,(ref_bases==7),1] = 1 #M
    one_hot_ref[:,(ref_bases==5),2] = 1 #K
    one_hot_ref[:,(ref_bases==5),3] = 1 #K

    flags_reads = np.zeros((N_reads,tensor_width,6)) # encoding of 6 flags, different for all reads

    #loop over flags of all reads
    for read_idx, (_,_,_,flag) in enumerate(aligned_reads):
        flags_reads[read_idx,:,0]=flag&0x2   #each segment properly aligned according to the aligner
        flags_reads[read_idx,:,1]=flag&0x8   #next segment unmapped
        flags_reads[read_idx,:,2]=flag&0x10  #SEQ being reverse complemented
        flags_reads[read_idx,:,3]=flag&0x20  #SEQ of the next segment in the template being reverse complemented
        flags_reads[read_idx,:,4]=flag&0x100 #secondary alignment
        flags_reads[read_idx,:,5]=flag&0x800 #supplementary alignment
        flags_reads[read_idx,:] = flags_reads[read_idx,:]>0 #to boolean

    tensor = {'one_hot_ref':one_hot_ref.astype(bool),
             'p_hot_reads':(p_hot_reads*1e4).astype(np.ushort),
            'flags_reads':flags_reads.astype(bool)}

    ref_support = ''.join(decode_bases(ref_bases)[variant_column_idx-10:variant_column_idx+11]) # reference sequence around the variant

    return tensor, ref_support, VAF0, DP0 #variant tensor, reference sequence around the variant, VAF and DP computed on non-truncated tensor
