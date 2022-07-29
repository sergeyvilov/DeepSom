# Construct variant tensors for each variant in a given VCF file.

import os
import sys
import time
import argparse

import pandas as pd

sys.path.append('utils/')

from get_tensors import get_tensors #function to generate tensors based on given VCF

parser = argparse.ArgumentParser("generate_tensors.py")

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

class dotdict(dict):
    '''
    Dictionary with dot.notation access to attributes
    '''

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

parser.add_argument("--vcf",                            help = "vcf or tsv file with variants", type = str, required = True)
parser.add_argument("--output_dir",                     help = "output dir name", type = str, required = True)
parser.add_argument("--bam_dir",                        help = "folder with bam files", type = str, required = True)
parser.add_argument("--refgen_fa",                      help = "reference genome FASTA file", type = str, required = True)
parser.add_argument("--Lbatch",                         help = "size of tensor batches", type = int, default = 32, required = False)
parser.add_argument("--tensor_width",                   help = "tensor width", type = int, default = 150, required = False)
parser.add_argument("--tensor_max_height",              help = "max tensor height", type = int, default = 70, required = False)
parser.add_argument("--tensor_crop_strategy",           help = "how to crop tensor when Nreads>tensor_max_height", type = str, default = 'topbottom', required = False)
parser.add_argument("--tensor_sort_by_variant",         help = "sort reads by base in the variant column", type = lambda x: bool(str2bool(x)), default = True, required = False)
parser.add_argument("--tensor_check_variant",           help = "perform basic checks for snps/indels", default = 'vaf_only', required = False) #'snps', 'indels', 'vaf_only' or 'None'
parser.add_argument("--replacement_csv",                help = "csv file with field chrom, pos, ref, alt when SNP mutation signatures are to be permuted", type=str, default = None, required = False) #'snps', 'indels', 'vaf_only' or 'None'


input_params = vars(parser.parse_args())

input_params = dotdict(input_params)

SIMULATE = 0 #don't create or delete images, just simulate

if not SIMULATE:
    os.makedirs(input_params.output_dir, exist_ok = True)

tensor_opts = dict() #parameters for the variant_to_tensor function

gen_params = dict() #parameters for the get_tensors function

for param,value in input_params.items():
    #from input parameters, separate parameters for get_tensors and variant_to_tensor functions
    if not param.startswith('tensor_'):
        gen_params[param] = value
    else:
        tensor_opts[param] = value

t0 = time.time()

variants_df = get_tensors(tensor_opts = tensor_opts, simulate=SIMULATE, **gen_params) #dataframe with annotations of processed variants

variants_df['vcf'] = os.path.basename(input_params.vcf) #vcf base name

variants_df.to_csv(os.path.join(gen_params['output_dir'], "variants.csv"), sep="\t", index=False)

t_exec = time.time() - t0 #total execution time

print(f"{gen_params['output_dir']}\nFinished successfully. Execution time: {t_exec//60:.0f}m {t_exec%60:.1f}s.")
print(f'{len(variants_df)} variants is created, distributed over {len(variants_df.batch_name.unique())} batches')
