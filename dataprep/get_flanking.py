################################################################################################################
# Get VAF and DP of flanking variants for each variant in a VCF file
#
#
# python get_flanking.py vcf_name min_gnomAD_AF flanking_region_length
#
# vcf_name: single-sample VCF file with all possible variants (Mutect2 output in tumor-only mode)
# VCF file should be annotated with gnomAD allele frequency (gnomAD_AF) in the INFO field
# min_gnomAD_AF: minimal gnomAD AF to consider that a variant is germline (to detect flanking variants)
# flanking_region_length: how many bases the flanking region spans
# to the left and (or to the righ) of a candidate variant
################################################################################################################

import numpy as np
import pandas as pd
import re
import os
import sys

MIN_AD = 5 #minimal AD for a flanking variant

def get_flanking(variants_pos, germline_df):
    '''
    Get flanking variants for each candidate variant position

    variants_pos: positions of candidate variants
    germline_df: (likely germline) variants, that can be flanking variants for positions in variants_pos
    '''

    germline_df.sort_values(by='pos', inplace=True)

    L = len(germline_df)

    AD = germline_df[['AD_ref','AD_alt']].values

    for variant_pos in variants_pos:

        min_left_idx = np.searchsorted(germline_df.pos, variant_pos-flanking_region_length)
        max_right_idx = np.searchsorted(germline_df.pos, variant_pos+flanking_region_length)
        variant_idx = np.searchsorted(germline_df.pos, variant_pos)

        AD_ref, AD_alt = AD[max(variant_idx-2,min_left_idx,0):variant_idx].sum(0) #combine ref and alt AD of at most 2 flanking variants to the left
        DP = AD_ref + AD_alt
        if DP>0:
            flanking_lVAF, flanking_lDP = AD_alt / DP, DP
        else:
            flanking_lVAF, flanking_lDP = -1, -1

        if variant_idx<L and germline_df.iloc[variant_idx].pos==variant_pos: #avoid overlap between candodate variant and its flanking variant
            variant_idx+=1

        AD_ref, AD_alt = AD[variant_idx:min(variant_idx+2,max_right_idx)].sum(0)  #combine ref and alt AD of at most 2 flanking variants to the right
        DP = AD_ref + AD_alt
        if DP>0:
            flanking_rVAF, flanking_rDP = AD_alt / DP, DP
        else:
            flanking_rVAF, flanking_rDP = -1, -1

        print(f'{round(flanking_lVAF,3)}|{flanking_lDP}|{round(flanking_rVAF,3)}|{flanking_rDP}')

min_gnomAD_AF = float(sys.argv[2])

flanking_region_length = int(sys.argv[3])

vcf = pd.read_csv(sys.argv[1], comment='#', sep='\t', usecols=[0,1,7,8,9], names=['chrom', 'pos', 'info', 'schema', 'format'], dtype={'chrom':str})

vcf['Filter'] = 'PASS' #all variants can potentially be flanking variants

#print('Filtering out all gnomAD variants with AF<',min_gnomAD_AF)

vcf['gnomAD_AF'] = vcf['info'].apply(lambda x: re.search('gnomAD_AF=([^;]*)',x).groups(1)[0] if 'gnomAD_AF' in x else 0).astype(float)

vcf.loc[vcf.gnomAD_AF < min_gnomAD_AF, 'Filter'] = 'no_germ_evidence'

#print('Filtering out all variants with AD<',MIN_AD)

AD_idx = vcf.schema.values[0].split(':').index('AD')

vcf['AD_ref'] = vcf.format.apply(lambda x:x.split(':')[AD_idx].split(',')[0]).astype(int)
vcf['AD_alt'] = vcf.format.apply(lambda x:x.split(':')[AD_idx].split(',')[1]).astype(int)

vcf.loc[(vcf.AD_ref<MIN_AD) | (vcf.AD_alt<MIN_AD), 'Filter'] = 'low_AD'

vcf = vcf[['chrom', 'pos', 'AD_ref', 'AD_alt', 'Filter']] #only variants with Filter=PASS can be flanking variants

#print('Looking for flanking variants')

for chrom in vcf.chrom.drop_duplicates():
    chrom_df = vcf[vcf.chrom==chrom]
    get_flanking(chrom_df.pos.values, chrom_df[chrom_df['Filter']=='PASS'])
