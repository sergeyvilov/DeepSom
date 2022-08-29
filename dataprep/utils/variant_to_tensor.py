import numpy as np
import re
import os

def is_valid_snp(ref, alt):
    '''
    Check if a snp variant is correct

    invalid snp:
    -REF or ALT allele is not in A,C,T or G
    -REF or ALT allele is longer than 1
    '''

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

def is_valid_indel(ref, alt):
    '''
    Check if an indel variant is correct

    invalid indel:
    -REF or ALT allele has a base which differs from A,C,T or G
    -Both REF and ALT allele are of length 1
    '''

    assert isinstance(ref,str), 'Ref allele is not a string'
    assert isinstance(alt,str), 'Alt allele is not a string'

    if len(ref)==len(alt)==1:
        raise Exception('Both ref and alt allele are of length 1, seems to be an SNP')

    if (set(ref)|set('ACTG'))!=set('ACTG'):
        raise Exception('Wrong base in ref allele')

    if (set(alt)|set('ACTG'))!=set('ACTG'):
        raise Exception('Wrong base in alt allele')

    return 0

def encode_bases(bases):
    '''
    Encode bases with digits
    '''
    nmap={'A':0,'C':1,'T':2,'G':3, 'N':4, 'K':5, '*':6, 'M':7} #asterix for deletion
    if len(bases)>1:
        return([nmap[n] for n in bases])
    else:
        return nmap[bases] #ACTGNK*M-->01234567

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
    Convert an ASCII quality character to the probability that the base is called incorrectly
    '''
    probability = 10 ** ((ord(qual_symbol) - 33) / -10)

    return probability

def compute_VAF_DP(reads, variant, vartype, varlen, ins_at_variant):
        '''
        Compute VAF and DP based on reads pileup
        '''
        if vartype=='ins':
            variant_column = [''.join(decode_bases(ins[1])) if len(ins)>0 else 'N' for ins in ins_at_variant]
            is_alt = [ins_seq==variant['alt'][1:] for ins_seq in variant_column]
        elif vartype=='del':
            variant_column = [read_seq[(variant['pos']-1)-read_pos+1:(variant['pos']-1)-read_pos+1+varlen] for read_pos, read_seq, *_ in reads]
            is_alt = [decode_bases(del_seq)==['*']*varlen for del_seq in variant_column]
        else:
            variant_column = [read_seq[(variant['pos']-1)-read_pos] for read_pos, read_seq, *_ in reads]
            is_alt = [decode_bases(alt)==variant['alt'] for alt in variant_column]

        AD_alt = sum(is_alt)
        DP = len(variant_column)

        return AD_alt/DP, DP, is_alt #VAF, DP, is ALT in read


def variant_to_tensor(variant, bam_file, ref_file,
                            tensor_width = 150, # tensor width
                            tensor_max_height = 70, #max tensor height
                            tensor_sort_by_variant = False, #sort reads by value in the variant column
                            tensor_check_variant = 'vaf_only', # perform basic checks for snps/indels: 'snps', 'indels' or 'vaf_only'
                            ):

    '''
    Collect reads for a given variant and transform them into a variant tensor.

    When reading the BAM file, consider only first MAX_RAW_READS reads.
    Discard reads with at least one flag from EXCLUDE_FLAGS.

    Use p-hot encoding for read bases and one-hot encoding for reference bases.

    Crop tensors whose height is above tensor_max_height, ensure VAF resolution of 1/tensor_max_height.

    Return tensor, reference sequence, VAF and DP before cropping.

    To learn more about encoding, see
    Friedman S. et al. Lean and deep models for more accurate filtering of SNP
    and INDEL variant calls //Bioinformatics. – 2020. – Т. 36. – №. 7. – С. 2060-2067.
    '''

    MAX_RAW_READS=1000 #maximum number of raw reads to consider for a variant

    EXCLUDE_FLAGS = 0x4|0x200|0x400 #exclude reads with flags "segment unmapped", "not passing filters", "PCR or optical duplicate"

    #basic check and filtering of invalid variants
    if tensor_check_variant=="snps":
        is_valid_snp(variant['ref'], variant['alt'])
    elif tensor_check_variant=="indels":
        is_valid_indel(variant['ref'], variant['alt'])

    if variant['ref']=='-' or len(variant['ref'])<len(variant['alt']):
        vartype='ins'
        varlen=len(variant['alt'])-1
    elif variant['alt']=='-' or len(variant['alt'])<len(variant['ref']):
        vartype='del'
        varlen=len(variant['ref'])-1
    else:
        vartype='snp'
        varlen=1

    raw_reads = []

    #collect all the reads around the candidate variant position
    for read in bam_file.fetch(variant['chrom'], variant['pos']-2, variant['pos']+2):
        if (read.flag&EXCLUDE_FLAGS==0 and read.pos<=variant['pos']-1 and read.reference_end>variant['pos']):
                raw_reads.append((read.pos,read.seq,read.qual,read.flag,read.cigartuples))
        if len(raw_reads)>=MAX_RAW_READS:#sometimes there are a lot of reads, we don't need all of them
            break

    #Align reads according to their CIGAR strings.
    #For each read its CIGAR string is analysed to place
    #read bases correctly, taking into account clips, insertions and deletions.

    aligned_reads = []

    #we shall collect insertions at the variant site only if the variant is an insertion

    ins_at_variant = [] #insertions at the variant site for all reads
    max_ins_length_at_variant = 0

    for read_idx, read in enumerate(raw_reads):

        pos,seq,qual,flag,cigartuples = read

        aligned_seq = []  #aligned read sequence
        aligned_qual = [] #aligned read qualities

        clipped = 0 #soft clip length at the beginning of the read
        is_ins_read = [] #positions of insertions in alinged read
        is_del_read = [] #positions of deletions in alinged read
        ins = []
        ins_len = 0

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
                if c==0: #if soft clip at the beginning
                    clipped=oplen #used to recalculate the insertion position
                c+=oplen
            elif optype==2 or optype==3 or optype==6: #deletion or padding
                aligned_seq.extend([encode_bases('*')]*oplen)
                aligned_qual.extend([0]*oplen)
                is_ins_read.extend([0]*oplen)
                is_del_read.extend([1]*oplen)#we mark ALL asterix as deletion (whereas insertions of any length we mark with only 1 symbol)
            elif optype==1:#insertion
                ins_pos=pos+c-clipped #absolute position AFTER which there is an insertion
                if ins_pos == variant['pos'] and vartype=='ins': #we collect insertions only if they are at the variant site
                    ins_len = oplen
                    ins = (ins_len, seq[c:c+oplen], qual[c:c+oplen])
                #if len(is_ins_read)>0:
                    #print(ins_pos)
                #    is_ins_read[-1]=1 #indicate that the NEXT operation is insertion - each insertion is marked only by 1 symbol, independently of its length
                c+=oplen
            else: #match or mismatch
                aligned_seq.extend(seq[c:c+oplen])
                aligned_qual.extend(qual[c:c+oplen])
                is_ins_read.extend([0]*oplen)
                is_del_read.extend([0]*oplen)
                c+=oplen

        aligned_reads.append((pos ,aligned_seq, aligned_qual, flag, is_ins_read, is_del_read))

        ins_at_variant.append(ins)
        max_ins_length_at_variant = max(max_ins_length_at_variant, ins_len)

    N_reads = len(aligned_reads)

    if vartype=='ins' and not  max_ins_length_at_variant:
        raise Exception('No insertion in the reads!')

    if not N_reads:
        raise Exception('No reads for this variant')

    VAF0,DP0,is_alt = compute_VAF_DP(aligned_reads, variant, vartype, varlen, ins_at_variant) #VAF and DP before cropping

    #print(f'Before cropping: Read depth (DP): {DP0}, VAF:{VAF0}')

    #if there are more reads than we can include in the tensor, we have to remove some of them

    if N_reads>tensor_max_height:
        #cropping
        #select N=tensor_max_height reads s.t. the VAF is preserved
        is_alt = np.array(is_alt)
        alt_indices = np.where(is_alt)[0] #indices of alt bases in the variant column
        ref_indices = np.where(is_alt==False)[0] #indices of ref bases in the variant column
        alt_indices_new = np.random.choice(alt_indices, int(VAF0*tensor_max_height), replace=False) #sample VAF0*tensor_max_height alt reads
        ref_indices_new = np.random.choice(ref_indices, min(tensor_max_height-len(alt_indices_new), len(ref_indices)), replace=False) #the remaining are ref reads
        chosen_indices = np.hstack((alt_indices_new, ref_indices_new))
        aligned_reads = [read for read_idx, read in enumerate(aligned_reads) if read_idx in chosen_indices] #choose only reads with given indices

    #OLD cropping strategies: VAF isn't preserved
    #if tensor_crop_strategy == 'center':
    #    #keep reads at the top and at the bottom, remove in the center
    #    aligned_reads = aligned_reads[:tensor_max_height//2] + aligned_reads[max(N_reads_tot-tensor_max_height//2,0):N_reads]
    #elif tensor_crop_strategy == 'topbottom':
    #    #keep reads in the center, remove at the top and at the bottom
    #    shift = max(N_reads//2-tensor_max_height//2,0)
    #    aligned_reads = aligned_reads[shift:shift+tensor_max_height]

    VAF,DP,is_alt = compute_VAF_DP(aligned_reads, variant, vartype, varlen, ins_at_variant) #VAF and DP after cropping

    #print(f'After cropping: Read depth (DP): {DP}, VAF:{VAF}')

    N_reads = len(aligned_reads)

    if tensor_check_variant in ('snps', 'indels', 'vaf_only') and VAF==0:
        #check if (after cropping) we still have the alternative allele in the variant column
        raise Exception('No reads with the alternative allele found in the variant column!')

    if tensor_sort_by_variant:
        #sort tensor by base in the variant column
        aligned_reads_sorted_tuple=sorted(zip(aligned_reads, ins_at_variant, is_alt),key=lambda x:x[2])
        aligned_reads, ins_at_variant = zip(*[read_data[:2] for read_data in aligned_reads_sorted_tuple])

    reads_im = np.zeros((N_reads,tensor_width,2)) #2 channels to encode the sequence and probability of each read base

    reads_im[:,:,0] = encode_bases('N') #bases for all reads, default: no data ('N')
    reads_im[:,:,1] = 1/4. #probability to have the corresponding basis, default:equal probability for all bases

    indels_chn = np.zeros((N_reads,tensor_width,2)) # encoding of indels

    #pileup reads
    variant_column_idx = tensor_width//2-(vartype in ('ins','del')) #variant column in the middle of the tensor, insertions/deletions shifted by 1bp to the left

    for read_idx, read in enumerate(aligned_reads):

        pos, seq, qual, flag, is_ins_read, is_del_read = read #absolute position, sequence, quality scores and flags of the read

        start_pos = pos-(variant['pos']-1-variant_column_idx) #relative position of the read in the tensor

        if start_pos<0: #left end of the read is beyond the defined tensor edge

            #reject data that's beyond the tensor edge
            seq = seq[-start_pos:]
            qual = qual[-start_pos:]
            is_ins_read = is_ins_read[-start_pos:]
            is_del_read = is_del_read[-start_pos:]
            start_pos = 0

        rlen = len(seq)

        reads_im[read_idx,start_pos:start_pos+rlen,0] = seq[:tensor_width-start_pos]
        reads_im[read_idx,start_pos:start_pos+rlen,1] = qual[:tensor_width-start_pos]

        indels_chn[read_idx,start_pos:start_pos+rlen,0] = is_ins_read[:tensor_width-start_pos]
        indels_chn[read_idx,start_pos:start_pos+rlen,1] = is_del_read[:tensor_width-start_pos]

    #if variant is an insertion then insert this insertion in the reads
    if vartype=='ins':

        reads_im[:,:,0] = np.hstack((reads_im[:,:variant_column_idx+1,0],np.ones((N_reads,max_ins_length_at_variant))*encode_bases('*'),reads_im[:,variant_column_idx+1:-max_ins_length_at_variant,0]))
        reads_im[:,:,1] = np.hstack((reads_im[:,:variant_column_idx+1,1],np.zeros((N_reads,max_ins_length_at_variant)),reads_im[:,variant_column_idx+1:-max_ins_length_at_variant,1]))

        indels_chn[:,:,0] = np.hstack((indels_chn[:,:variant_column_idx+1,0],np.zeros((N_reads,max_ins_length_at_variant)),indels_chn[:,variant_column_idx+1:-max_ins_length_at_variant,0]))
        indels_chn[:,:,1] = np.hstack((indels_chn[:,:variant_column_idx+1,1],np.ones((N_reads,max_ins_length_at_variant)),indels_chn[:,variant_column_idx+1:-max_ins_length_at_variant,1]))

        for read_idx,ins in enumerate(ins_at_variant):

            if ins:

                ins_len,seq,qual = ins

                reads_im[read_idx,variant_column_idx+1:variant_column_idx+1+ins_len,0] = seq[:variant_column_idx]
                reads_im[read_idx,variant_column_idx+1:variant_column_idx+1+ins_len,1] = qual[:variant_column_idx]

                indels_chn[read_idx,variant_column_idx+1:variant_column_idx+1+ins_len,0] = 1#insertions
                indels_chn[read_idx,variant_column_idx+1:variant_column_idx+1+ins_len,1] = 0#fill deletions with 0 where there's actually an insertion

    def get_ref_bases(variant, variant_is_insertion, max_ins_length_at_variant):

        '''
        Get reference sequence around the variant positon, variant being roughly in the center
        '''

        ref_bases = ref_file.fetch(variant['chrom'], variant['refpos']-variant_column_idx-1, variant['refpos']-variant_column_idx+tensor_width-1) #reference bases around the variant position

        ref_bases = ref_bases.upper() #ignore strand information

        if variant_is_insertion: #insert deletions in the reference sequence if variant is an insertion
            ref_bases = ref_bases[:variant_column_idx+1]+'*'*max_ins_length_at_variant+ref_bases[variant_column_idx+1:-max_ins_length_at_variant]

        L_ref = len(variant['ref'])

        if ref_bases[variant_column_idx:variant_column_idx+L_ref]!=variant['ref']:
            raise Exception(f'Variant reference allele not found in the right position in the reference bases string: {ref_bases} {variant_column_idx} {variant["refpos"]-variant_column_idx}!')

        ref_bases = np.array(encode_bases(ref_bases)) # letters to digits

        return ref_bases

    ref_bases = get_ref_bases(variant, vartype=='ins', max_ins_length_at_variant)

    p_hot_reads = np.zeros((N_reads,tensor_width,4)) # p-hot encoding of read bases probabilities: each channel gives the probability of the corresponding base (ACTG)

    for basis_idx in range(4):
        p_hot_reads[:,:,basis_idx] = np.where(reads_im[:,:,0]==basis_idx, reads_im[:,:,1], (1.-reads_im[:,:,1])/3.)

    del_row, del_col = np.where(reads_im[:,:,0]==encode_bases('*')) #deletions
    p_hot_reads[del_row, del_col, :] = 0.

    M_row, M_col = np.where(reads_im[:,:,0]==encode_bases('M')) #M: either A or C, each with probability 0.5
    p_hot_reads[M_row, M_col, 0] = 0.5
    p_hot_reads[M_row, M_col, 1] = 0.5

    K_row, K_col = np.where(reads_im[:,:,0]==encode_bases('K')) #K: either G or T, each with probability 0.5
    p_hot_reads[K_row, K_col, 2] = 0.5
    p_hot_reads[K_row, K_col, 3] = 0.5

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
    for read_idx, (_,_,_,flag,_,_) in enumerate(aligned_reads):
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

    ref_support = ''.join(decode_bases(ref_bases)[variant_column_idx-30:variant_column_idx+31]) # reference sequence around the variant

    return tensor, ref_support, VAF0, DP0 #variant tensor, reference sequence around the variant, VAF and DP computed befor cropping the tensor
