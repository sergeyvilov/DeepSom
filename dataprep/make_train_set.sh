#!/bin/bash

#######################################################
#
#Select required number of variants for CNN training
#
#Input: train.vcf.gz
#Output: train.selected.tsv
#######################################################

conda activate dataprep

min_DP=1 #minimum coverage at the variant site

#Required number of variants in the output file
N_train_snps_negative=60000
N_train_indels_negative=13000 #x1.3 because many non-somatic indels will be filtered out during tensor generation
N_train_snps_somatic=60000
N_train_indels_somatic=10000

extract_variants () {

  input_vcf_gz=$1
  N_variants_out=$2
  variant_class=$3 #'somatic' or 'non-somatic'
  variant_type=$4 #'snps' or 'indels'

  if [ $variant_class = "somatic" ];then
    filtering_string="(FORMAT/DP>$min_DP)&&(SOMATIC=1)"
  else
    filtering_string="(FORMAT/DP>$min_DP)&&(SOMATIC=0)"
  fi

  prefiltered_variants_tsv=prefiltered_vatiants.tsv

  bcftools view -H $input_vcf_gz -v "$variant_type" -i "$filtering_string" > $prefiltered_variants_tsv #select variants of given type and minimum read depth $min_DP

  N_variants=$(wc -l $prefiltered_variants_tsv|cut -d" " -f1) #number of available variants

  if [ $N_variants -lt $N_variants_out ];then
    #upsampling is necessary
    echo "Insufficient number of train varints available ($N_variants instead of $N_variants_out), upsampling..."
    shuf -r -n $N_variants_out $prefiltered_variants_tsv #shuffle and select the required number of variants, with replacement
  else
    #upsampling is not necessary
    shuf -n $N_variants_out $prefiltered_variants_tsv #shuffle and select the required number of variants, without replacement
  fi

}

printf "\n*Selecting $N_train_snps_negative negative train snps\n"
extract_variants train.vcf.gz $N_train_snps_negative non-somatic snps > negative_train_snps.temp.tsv
printf "\n*Selecting $N_train_indels_negative negative train indels\n"
extract_variants train.vcf.gz $N_train_indels_negative non-somatic indels > negative_train_indels.temp.tsv
printf "\n*Selecting $N_train_snps_somatic somatic train snps\n"
extract_variants train.vcf.gz $N_train_snps_somatic somatic snps > somatic_train_snps.temp.tsv
printf "\n*Selecting $N_train_indels_somatic somatic train indels\n"
extract_variants train.vcf.gz $N_train_indels_somatic somatic indels  > somatic_train_indels.temp.tsv

cat negative_train_snps.temp.tsv somatic_train_snps.temp.tsv negative_train_indels.temp.tsv somatic_train_indels.temp.tsv|shuf > train.selected.tsv

#Split output file into files of $variants_per_split variants each, to generate tensors in parallel
#
#printf "\n*Merging train tsv\n"
#cat negative_train_snps.temp.tsv somatic_train_snps.temp.tsv|shuf > train_snps.temp.tsv
#cat negative_train_indels.temp.tsv somatic_train_indels.temp.tsv|shuf > train_indels.temp.tsv
#
#
#variants_per_split=5000
#
#mkdir -p train/
#
#printf "\n*Making short variants lists, $variants_per_split variants each \n"
#cat train_snps.temp.tsv|awk -v variants_per_split=$variants_per_split 'BEGIN{output_idx=0}{if (NR%variants_per_split==0){output_idx=NR}{print $0>"train/train_snps_"output_idx".tsv"}}'
#cat train_indels.temp.tsv|awk -v variants_per_split=$variants_per_split 'BEGIN{output_idx=0}{if (NR%variants_per_split==0){output_idx=NR}{print $0>"train/train_indels_"output_idx".tsv"}}'

rm *temp*

echo "Done"
