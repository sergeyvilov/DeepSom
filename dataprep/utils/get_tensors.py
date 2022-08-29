import pickle
import os
from typing import *
import re
import pysam
import random
import warnings

import pandas as pd

from variant_to_tensor import variant_to_tensor #function to generate a variant tensor

def dump_batch(batch_tensors, batch_info, output_dir, batch_name, simulate=False):
    '''
    Write a batch of tensors on the disk
    '''
    if not simulate:
        os.makedirs(output_dir, exist_ok = True)
        with open(os.path.join(output_dir, batch_name), 'wb') as f:
            for tensor, info in zip(batch_tensors, batch_info):
                pickle.dump([tensor,info], f)

def get_tensors(vcf :str,                             #full path to a VCF/TSV file with the variants
               bam_dir: str,                          #directory with all BAM files
               output_dir :str,                       #output dir for imgb batches
               refgen_fa :str,                        #reference genome FASTA file
               tensor_opts :Dict,                     #options for variant tensor encoding
               Lbatch :Optional[int] = 1,             #how many tensors to put in each batch
               simulate :Optional[bool] = False,      #simulate workflow, don't save tensors
               replacement_csv :Optional[str] = None, #replace mutational signatures by randomly choosing those from this file
             ):
    '''
    Construct imgb batches based on variants in the input vcf (tsv) file.

    The header and all columns following the INFO column in the VCF file are ignored.

    For each variant a sample BAM file should be provided.
    BAM file name must be indicated as 'BAM=bam_file_name.bam' (without the folder path) in the INFO field.

    For train/evaluation all somatic variants should be labelled as 'SOMATIC' in the INFO field.

    Each imgb batch consists of Lbatch variant tensors.
    Depending on the simulate option the batches are saved to the disk.

    Each variant tensor is coupled with variant annotations.
    These annotations include original vcf file name (vcf),
    variant record index in this vcf file (record_idx), chromosome name(chrom),
    position (pos), reference allele (ref), alternative allele (alt),
    imgb batch name (batch_name), position of the variant in the imgb batch (imgb_index).
    We also add variant allele fraction (VAF0), read depth (DP0) and the reference sequence
    around the variant (refseq) which we obtain during tensor generation. We also keep the
    INFO field from the original VCF.

    Variant annotations are also added to the variants_df dataframe.
    To speed up processing, they are first collected in variants_list and added to
    the variants_df only when 1000 variants are accumulated.

    Tensor options (width, height etc...) are defined in the tensor_opts dictionary.
    See the variant_to_tensor function to learn more about tensor options.
    '''

    BLOCK_LENGTH = 5000 #chunk of vcf file to read at once

    variants_df = pd.DataFrame(columns=["vcf", "record_idx", "chrom", "pos", "ref", "alt", "VAF0", "DP0", "tensor_height", "batch_name", "imgb_index", "remarks"]) # DataFrame for variant annotations

    vcf_basename = os.path.basename(vcf) #name w/o path
    vcf_short_name = re.sub('(\.vcf|\.tsv)(\.gz){0,1}$','', vcf_basename) #remove extension

    vcf_in = pd.read_csv(vcf, comment='#', sep='\t', names=['chrom', 'pos', 'id', 'ref', 'alt', 'qual', 'filter', 'info'], dtype={'chrom':str}, usecols = [col_idx for col_idx in range(8)])

    vcf_in['bam'] = vcf_in['info'].apply(lambda x: re.search('BAM=([^;]*)',x).groups(1)[0] if 'BAM=' in x else None)

    #check if all BAM files exist
    for bam_file_name in vcf_in.bam.drop_duplicates():

            bam_path = os.path.join(bam_dir, bam_file_name)

            if not os.path.isfile(bam_path):
                warnings.warn(f'BAM file not found: {bam_path}')

    #vcf_in['true_label'] = vcf_in['info'].apply(lambda x: 1 if 'SOMATIC' in x else 0).astype(int)
    #vcf_in['vartype'] = vcf_in[['ref','alt']].apply(lambda x: 'SNP' if  len(x.ref)==len(x.alt) else 'INDEL', axis=1)

    vcf_in = vcf_in.sample(frac=1., random_state=hash(vcf)%2**32) #shuffle input vcf

    if replacement_csv:
        replacement_df = pd.read_csv(replacement_csv, names=['chrom', 'refpos', 'ref', 'alt'], dtype={'chrom':str, 'refpos':int})

    ref_file = pysam.FastaFile(refgen_fa) #open reference genome FASTA

    for block_start in range(0, len(vcf_in), BLOCK_LENGTH):

        #we read the VCF file by blocks of length BLOCK_LENGTH
        #for each block, we accumulate tensors, shuffle them, and distribute over imgb batches

        block_variants = [] #variants in current block

        block_end = min(block_start + BLOCK_LENGTH, len(vcf_in))

        block_vcf = vcf_in.iloc[block_start:block_end] #chunk of VCF corresponding to the current block

        #loop over BAM files
        for bam_file_name in block_vcf.bam.drop_duplicates():

            bam_path = os.path.join(bam_dir, bam_file_name)

            if not os.path.isfile(bam_path):
                continue

            bam_file = pysam.AlignmentFile(bam_path, "rb" ) #open the BAM file

            for record_idx, rec in block_vcf[block_vcf.bam==bam_file_name].iterrows():

                variant_info = {'pos':rec.pos, 'refpos':rec.pos, 'chrom':rec.chrom, 'ref':rec.ref, 'alt':rec.alt,
                    'info': rec['info'].rstrip(';'), 'vcf': vcf_basename, 'record_idx':record_idx}

                if 'POS_Build36' in rec['info']: #for some samples in TCGA-LAML
                    variant_info['pos'] = int(re.search('POS_Build36=([^;]*)',rec['info']).groups(1)[0])

                if replacement_csv:
                    #replace mutational context
                    chrom, refpos, ref, alt = replacement_df.sample(n=1).values[0]
                    replacement_variant = {'chrom':chrom, 'refpos':refpos, 'ref':ref, 'alt':alt}

                else:

                    replacement_variant = None

                #
                # variant_annotations = {}
                #
                # for ann_name in ['GERMLINE']:
                #     if ann_name in rec.info.keys():
                #         variant_annotations[ann_name] = rec.info.get(ann_name)
                #     else:
                #         variant_annotations[ann_name] = None
                #
                # variant_info.update(variant_annotations)

                try:

                    #get a tensor for the current variant
                    variant_tensor, ref_support, VAF0, DP0 = variant_to_tensor(variant_info, bam_file, ref_file, replacement_variant=replacement_variant,
                         **tensor_opts) #variant tensor, reference sequence around the variant, VAF and DP computed on non-truncated tensor

                    tensor_height = variant_tensor['p_hot_reads'].shape[0] #tensor height after cropping (can be smaller than DP)

                    variant_info.update({'VAF0': round(VAF0,2), 'DP0': DP0, 'refseq': ref_support,'tensor_height': tensor_height, 'remarks': 'success'})

                except Exception as error_msg:

                   print('-------------------------------------------------')
                   print('Exception occured while creating a variant tensor')
                   print('Variant:\n', variant_info)
                   print('Reference FASTA file:\n', refgen_fa)
                   print('BAM file:\n', bam_path)
                   print('Error message:\n', error_msg)

                   variant_info['remarks'] = error_msg

                   variant_tensor = None

                for key in ('chrom', 'refpos', 'ref', 'alt'):
                    variant_info['replaced_' + key] = replacement_variant[key] if replacement_variant else None

                block_variants.append((variant_tensor, variant_info))

            bam_file.close()

        random.shuffle(block_variants)

        #distribute block variants over imgb batches

        batch_tensors, batch_info, imgb_index = [], [], 0

        for variant_tensor, variant_info in block_variants:

            if variant_tensor!=None: #if a tensor was created

                if imgb_index==0: #1st variant in the imgb batch
                    batch_name = f'{vcf_short_name}_{variant_info["record_idx"]}.imgb'

                variant_info.update({'imgb_index':imgb_index, 'batch_name':batch_name})

                batch_tensors.append(variant_tensor)
                batch_info.append(variant_info)

                imgb_index += 1

                if imgb_index==Lbatch:
                    dump_batch(batch_tensors, batch_info, output_dir, batch_name, simulate=simulate)
                    batch_tensors, batch_info, imgb_index = [], [], 0

        _, block_info = zip(*block_variants)

        variants_df = variants_df.append(list(block_info), ignore_index=True)

        if imgb_index>0:
            dump_batch(batch_tensors, batch_info, output_dir, batch_name, simulate=simulate)

    ref_file.close()

    return variants_df
