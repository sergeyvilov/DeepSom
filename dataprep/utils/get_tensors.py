import pickle
import os
from typing import *
import re
import pysam
import random

import pandas as pd

#import pysam #library for reading VCF files

from variant_to_tensor import variant_to_tensor #function to form a tensor out of a variant

def dump_batch(batch_tensors, batch_info, output_dir, batch_name, simulate=False):
    '''
    Write a batch of tensors on the disk
    '''
    if not simulate:
        os.makedirs(output_dir, exist_ok = True)
        with open(os.path.join(output_dir, batch_name), 'wb') as f:
            pickle.dump({'images':batch_tensors, 'info':batch_info}, f)

def get_tensors(vcf :str,                             #full path to a VCF file with the variants
               bam_dir: str,                          #directory with corresponding BAM files
               output_dir :str,                       #output dir for tensor batches
               refgen_fa :str,                        #reference genome FASTA file
               tensor_opts :Dict,                     #options for variant tensor encoding
               Lbatch :Optional[int] = 1,             #how many tensors put in each batch
               simulate :Optional[bool] = False,      #simulate workflow, don't dump tensors
               replacement_csv :Optional[str] = None, #randomly replace mutational signatures by sampling variants from this file
             ):
    '''
    Construct imgb batches based on variants in the input  vcf (tsv) file.

    The header and all columns following the INFO column in the vcf file are ignored.

    For each variant a sample BAM file is required.
    BAM file must be added as a record BAM=bam_file_name.bam in the VCF INFO field (without the folder path).

    All somatic variants in the vcf file should be labelled as "SOMATIC" in the INFO field.

    Each imgb batch consists of Lbatch variant tensors.
    Depending on the simulate option the batches are saved to the disk.
    To avoid file system issues, we distribute batches over several subfolders in the output_dir.

    To keep record of variants created, we add variant annotations to the 'info' field of
    each imgb batch. These annotations include original vcf file name (vcf),
    variant record index in the vcf file (record_idx), chrom, pos, ref, alt, true_label,
    batch name, index of the variant in batch (imgb_index), batch subfolder (subdir).
    We also add variant allele fraction (VAF0), read depth (DP0) and the reference sequence
    around the variant (refseq) that we obtain during tensor generation. The
    INFO field from the original vcf is also kept.

    Variant annotations are also added to the variants_df dataframe.
    To speed up processing, they are first accumulated in variants_list and added to
    the variants_df only when 1000 variants are accumulated.

    Tensor options (width, height etc...) are defined in the tensor_opts dictionary.
    See the variant_to_tensor function to learn more about tensor options.
    '''

    BLOCK_LENGTH = 1000

    variants_df = pd.DataFrame(columns=["vcf", "record_idx", "chrom", "pos", "ref", "alt", "VAF0", "DP0", "tensor_height", "batch_name", "imgb_index", "subdir", "remarks"]) # DataFrame for variant annotations

    vcf_basename = os.path.basename(vcf) #name w/o path
    vcf_short_name = re.sub('(\.vcf|\.tsv)(\.gz){0,1}$','', vcf_basename) #remove extension

    #iterate over the records of the vcf file

    vcf_in = pd.read_csv(vcf, comment='#', sep='\t', names=['chrom', 'pos', 'id', 'ref', 'alt', 'qual', 'filter', 'info'], dtype={'chrom':str}, usecols = [i for i in range(8)])

    vcf_in['bam'] = vcf_in['info'].apply(lambda x: re.search('BAM=([^;]*)',x).groups(1)[0] if 'BAM=' in x else None)

    for bam_file_name in vcf_in.bam.drop_duplicates():

            bam_path = os.path.join(bam_dir, bam_file_name)

            if not os.path.isfile(bam_path):
                raise Exception('BAM file not found: ', bam_path)

    vcf_in['true_label'] = vcf_in['info'].apply(lambda x: 1 if 'SOMATIC' in x else 0).astype(int)

    vcf_in = vcf_in.sample(frac=1., random_state=1) #shuffle input vcf

    if replacement_csv:
        replacement_df = pd.read_csv(replacement_csv, names=['chrom', 'refpos', 'ref', 'alt'], dtype={'chrom':str, 'refpos':int})

    #vcf_in['vartype'] = vcf_in[['ref','alt']].apply(lambda x: 'SNP' if  len(x.ref)==len(x.alt) else 'INDEL', axis=1)

    total_batches = 0

    ref_file = pysam.FastaFile(refgen_fa) #open reference genome FASTA

    random.seed(1)

    for block_start in range(0, len(vcf_in), BLOCK_LENGTH):

        block_variants = []

        block_end = min(block_start + BLOCK_LENGTH, len(vcf_in))

        block_vcf = vcf_in.iloc[block_start:block_end]

        for bam_file_name in block_vcf.bam.drop_duplicates():

            bam_path = os.path.join(bam_dir, bam_file_name)

            bam_file = pysam.AlignmentFile(bam_path, "rb" ) #open the BAM file

            for record_idx, rec in block_vcf[block_vcf.bam==bam_file_name].iterrows():

                variant = {'pos':rec.pos, 'refpos':rec.pos, 'chrom':rec.chrom, 'ref':rec.ref, 'alt':rec.alt,
                'true_label':rec.true_label, 'info': rec['info'].rstrip(';'), 'vcf': vcf_basename, 'record_idx':record_idx}

                if 'POS_Build36' in rec['info']:
                    variant['pos'] = int(re.search('POS_Build36=([^;]*)',rec['info']).groups(1)[0])

                if replacement_csv:

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
                # variant.update(variant_annotations)

                try:

                    #get a tensor variant tensor for the current variant
                    variant_tensor, ref_support, VAF0, DP0 = variant_to_tensor(variant, bam_file, ref_file, replacement_variant=replacement_variant,
                         **tensor_opts) #variant tensor, reference sequence around the variant, VAF and DP computed on non-truncated tensor

                    tensor_height = variant_tensor['p_hot_reads'].shape[0] #tensor height after cropping (can be smaller than DP)

                    variant.update({'VAF0': round(VAF0,2), 'DP0': DP0, 'refseq': ref_support,'tensor_height': tensor_height, 'remarks': 'success'})

                except Exception as error_msg:

                   print('-------------------------------------------------')
                   print('Exception occured while creating a variant tensor')
                   print('Variant:\n', variant)
                   print('Reference FASTA file:\n', refgen_fa)
                   print('BAM file:\n', bam_path)
                   print('Error message:\n', error_msg)

                   variant['remarks'] = error_msg

                   variant_tensor = None

                for key in ('chrom', 'refpos', 'ref', 'alt'):
                    variant['replaced_' + key] = replacement_variant[key] if replacement_variant else None

                block_variants.append((variant_tensor, variant))

            bam_file.close()

        random.shuffle(block_variants)

        variants_list = []

        batch_tensors, batch_info, imgb_index = [], [], 0

        for variant_tensor, variant in block_variants:

            if variant_tensor!=None:

                if imgb_index==0:
                    batch_name = f'{vcf_short_name}_{variant["record_idx"]}.imgb'

                batch_subdir = str(total_batches//100)

                variant.update({'imgb_index':imgb_index, 'batch_name':batch_name, 'subdir':batch_subdir})

                batch_tensors.append(variant_tensor)
                batch_info.append(variant)

                imgb_index += 1

                if imgb_index==Lbatch:
                    dump_batch(batch_tensors, batch_info, os.path.join(output_dir,  batch_subdir), batch_name, simulate=simulate)
                    batch_tensors, batch_info, imgb_index = [], [], 0
                    total_batches += 1

            variants_list.append(variant)

        variants_df = variants_df.append(variants_list, ignore_index=True)

        if imgb_index>0:
            dump_batch(batch_tensors, batch_info, os.path.join(output_dir,  batch_subdir), batch_name, simulate=simulate)

    ref_file.close()

    return variants_df
