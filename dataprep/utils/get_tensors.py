import pickle
import os
from typing import *

import pandas as pd

import pysam #library for reading VCF files

from variant_to_tensor import variant_to_tensor #function to form a tensor out of a variant

def dump_batch(batch, info, batch_path, simulate=False):
    '''
    Write a batch of tensors on the disk
    '''
    #print(batch_path)

    if not simulate:
        with open(batch_path, 'wb') as f:
            pickle.dump({'images':batch, 'info':info},f)

def get_tensors(vcf :str,                             #full path to a VCF file with the variants
               bam_dir: str,                          #directory with corresponding BAM files
               output_dir :str,                       #output dir for tensor batches
               refgen_fa :str,                        #reference genome FASTA file
               tensor_opts :Dict,                     #options for variant tensor encoding
               Lbatch :Optional[int] = 1,             #how many tensors put in each batch
               chrom :Optional[str] = None,           #chromosome name
               chrom_start :Optional[int] = None,     #start position in the chromosome
               chrom_stop :Optional[int] = None,      #stop position in the chromosome
               max_variants :Optional[int] = None,    #stop when this number of variants is reached
               bam_matching_csv :Optional[str] = '',  #matching table between BAM sample name and BAM file name
               simulate :Optional[bool] = False,      #simulate workflow, don't dump tensors
             ):
    '''
    Create a pileup tensor for each variant in the given VCF file.

    For each variant a sample BAM file is required.
    BAM file name can be encoded directly as a record BAM=bam_file_name.bam in the VCF INFO field (without the path).
    Otherwise, it is inferred from the sample name in the VCF file using bam_matching_csv.

    Tensors a packed in batches of size Lbatch.
    Depending on the global simulate value the batches are saved to the disk.
    To avoid file system issues, we distribute batches into subfolders in the output_dir.

    To keep record of variants created, variant annotations (DP, VAF etc...) are added to the variants_df dataframe.
    To speed up processing, they are first accumulated in variants_list and added to the variants_df only when 1000 variants are accumulated.

    Tensor options (width, height etc...) are defined in the tensor_opts dictionary.
    See the variant_to_tensor function to learn more about tensor options.
    '''

    tensors_per_subdir = 100*Lbatch #maximum tensors per subdir

    variants_df = pd.DataFrame(columns=["vcf", "chrom_record_idx", "chrom", "pos", "ref", "alt", "BAM", "VAF", "DP", "tensor_height", "batch_name", "imgb_index", "subdir"]) # DataFrame for variant annotations

    if not simulate:
        os.makedirs(output_dir, exist_ok=True)

    if bam_matching_csv:
        #matching table between BAM sample name and BAM file name
        #otherwise, the INFO filed of the VCF file should have the BAM=bam_file_name.bam record
        bam_matching = pd.read_csv(bam_matching_csv, names=['BAM_sample', 'BAM_file'], squeeze=True, index_col=0)
        bam_matching = bam_matching.apply(lambda x:x.replace('.bam','')+'.bam')

    vcf_in = pysam.VariantFile(vcf) #open the VCF file

    vcf_basename = os.path.basename(vcf).replace('.vcf.gz','')

    all_samples = list(vcf_in.header.samples) #extract BAM sample names from the VCF header

    variants_batch = [] #current batch of tensors

    variants_list = []  #we will first accumulate variant annotations in a list and then add this list to the data frame

    N_variants_added = 0 #total number of variants added

    #iterate over the records of the vcf file
    for chrom_record_idx, rec in enumerate(iter(vcf_in.fetch(contig = chrom))):

        if (chrom_start!=None and rec.pos<chrom_start) or (chrom_stop!=None and rec.pos>chrom_stop):
            continue

        if chrom_record_idx%tensors_per_subdir==0:
            #switch to a new subdir if the current one already has enough batches
            batch_subdir = str(chrom_record_idx//tensors_per_subdir)
            os.makedirs(os.path.join(output_dir, batch_subdir), exist_ok = True)

        if max_variants and N_variants_added >= max_variants:
            break

        #in a VCF file we have BAM sample names and we need the names of corresponding BAM files
        if bam_matching_csv:
            #get the file name from the matching table
            bam_sample_names = [s for s in all_samples if rec.samples[s]['GT']!=(None,None)]
            bam_file_names = bam_matching.loc[bam_sample_names]
        else:
            #otherwise, use BAM file name from the VCF record
            bam_file_name = rec.info.get('BAM')[0].replace('.bam','')+'.bam' #when the BAM file name is defined in the INFO field
            bam_file_names = [bam_file_name] #for compatibility


        #loop over all BAM files that have this variant
        for bam_file_name in bam_file_names:

                bam_path = os.path.join(bam_dir, bam_file_name) #full path to the BAM file

                variant = {'pos':rec.pos, 'refpos':rec.pos, 'chrom':rec.chrom, 'ref':rec.ref, 'alt':rec.alts[0]}

                try:

                    #get a tensor variant tensor for the current variant
                    variant_tensor, ref_support, VAF, DP = variant_to_tensor(variant, refgen_fa, bam_path, check_variant="snps",
                         **tensor_opts) #variant tensor, reference sequence around the variant, VAF and DP computed on non-truncated tensor

                except Exception as exc:

                   print('-------------------------------------------------')
                   print('Exception occured while creating a variant tensor')
                   print('Variant:\n', variant)
                   print('Reference FASTA file:\n', refgen_fa)
                   print('BAM file:\n', bam_path)
                   print('Error message:\n', exc)

                   continue

                variants_batch.append(variant_tensor) #add current variant to the batch

                tensor_height = variant_tensor['p_hot_reads'].shape[0] #tensor height after cropping (can be smaller than DP)

                variant_record = {
                     'vcf': vcf_basename,
                     'chrom_record_idx':chrom_record_idx,
                     'subdir': batch_subdir,
                     'BAM': bam_file_name,
                     'VAF': VAF,
                     'DP': DP,
                     'tensor_height':tensor_height,
                    }

                variant_record.update(variant)

                variants_list.append(variant_record)

                N_variants_added += 1

                if N_variants_added%Lbatch == 0:

                    #save the batch to the disk when it is full

                    batch_name = f'{vcf_basename}_{variants_list[-Lbatch]["chrom_record_idx"]}.imgb' #batch name: VCF record index (within the given chrom) of the 1st variant in the batch

                    for i in range(-Lbatch,0):
                        variants_list[i]['batch_name']=batch_name #mark batch name in the variants list
                        variants_list[i]['imgb_index']=i+Lbatch #tensor index in imgb batch


                    if not simulate:
                        #save batch to the disk
                        dump_batch(variants_batch, variants_list[-Lbatch:], os.path.join(*[output_dir, batch_subdir, batch_name]), simulate=simulate)

                    variants_batch = [] #empty current batch

                    if  len(variants_list)>1000:
                        #add variants_list to variants_df every 1000 tensors
                        variants_df = variants_df.append(variants_list, ignore_index=True)
                        variants_list = []

    N_batch = len(variants_batch)

    if N_batch:

        batch_name = f"{vcf_basename}_{variants_list[-N_batch]['chrom_record_idx']}.imgb" #batch name: VCF record index (within the given chrom) of the 1st variant in the batch

        for i in range(-N_batch,0):
            variants_list[i]['batch_name']=batch_name #mark batch name in the variants list
            variants_list[i]['imgb_index']=i+N_batch #tensor index in imgb batch

        if not simulate:
            #save batch to the disk
            dump_batch(variants_batch, variants_list[-N_batch:], os.path.join(*[output_dir,  batch_subdir, batch_name]), simulate=simulate)

    variants_df = variants_df.append(variants_list, ignore_index=True)

    return variants_df
