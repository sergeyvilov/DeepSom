import numpy as np
import pandas as pd
import re
import os
import sys

MIN_AD = 5

def get_flanking(variants_pos, negative_chrom_df):

    negative_chrom_df.sort_values(by='pos', inplace=True)

    L = len(negative_chrom_df)

    AD = negative_chrom_df[['AD_ref','AD_alt']].values

    for variant_pos in variants_pos:

        min_left_idx = np.searchsorted(negative_chrom_df.pos, variant_pos-flanking_region_length)
        max_right_idx = np.searchsorted(negative_chrom_df.pos, variant_pos+flanking_region_length)
        variant_idx = np.searchsorted(negative_chrom_df.pos, variant_pos)

        AD_ref, AD_alt = AD[max(variant_idx-2,min_left_idx,0):variant_idx].sum(0)
        DP = AD_ref + AD_alt
        if DP>0:
            flanking_lVAF, flanking_lDP = AD_alt / DP, DP
        else:
            flanking_lVAF, flanking_lDP = -1, -1

        if variant_idx<L and negative_chrom_df.iloc[variant_idx].pos==variant_pos:
            variant_idx+=1

        AD_ref, AD_alt = AD[variant_idx:min(variant_idx+2,max_right_idx)].sum(0)
        DP = AD_ref + AD_alt
        if DP>0:
            flanking_rVAF, flanking_rDP = AD_alt / DP, DP
        else:
            flanking_rVAF, flanking_rDP = -1, -1

        print(f'{round(flanking_lVAF,3)}|{flanking_lDP}|{round(flanking_rVAF,3)}|{flanking_rDP}')

min_gnomAD_AF = float(sys.argv[2])

flanking_region_length = int(sys.argv[3])

vcf = pd.read_csv(sys.argv[1], comment='#', sep='\t', usecols=[0,1,7,8,9], names=['chrom', 'pos', 'info', 'schema', 'format'], dtype={'chrom':str})

vcf['Filter'] = 'PASS'

#print('Filtering out all gnomAD variants with AF<',min_gnomAD_AF)

vcf['gnomAD_AF'] = vcf['info'].apply(lambda x: re.search('gnomAD_AF=([^;]*)',x).groups(1)[0] if 'gnomAD_AF' in x else 0).astype(float)

vcf.loc[vcf.gnomAD_AF < min_gnomAD_AF, 'Filter'] = 'no_germ_evidence'

#print('Filtering out all variants with AD<',MIN_AD)

AD_idx = vcf.schema.values[0].split(':').index('AD')

vcf['AD_ref'] = vcf.format.apply(lambda x:x.split(':')[AD_idx].split(',')[0]).astype(int)
vcf['AD_alt'] = vcf.format.apply(lambda x:x.split(':')[AD_idx].split(',')[1]).astype(int)

vcf.loc[(vcf.AD_ref<MIN_AD) | (vcf.AD_alt<MIN_AD), 'Filter'] = 'low_AD'

vcf = vcf[['chrom', 'pos', 'AD_ref', 'AD_alt', 'Filter']]

#print('Looking for flanking variants')

for chrom in vcf.chrom.drop_duplicates():
    chrom_df = vcf[vcf.chrom==chrom]
    get_flanking(chrom_df.pos.values, chrom_df[chrom_df['Filter']=='PASS'])
