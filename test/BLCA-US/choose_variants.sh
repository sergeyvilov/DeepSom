#!/bin/bash

#choose well-separable variants for demo

#model predictions
output_vcf='/lustre/groups/epigenereg01/workspace/projects/vale/nnc_logs/BLCA-US/train_test_flanking/gnomAD_thr_0/dropout_0.5/20220901/test_whole_on_self/20220725/predictions/final_predictions.vcf.gz'

snp_thr=0.92 #cnn output threshold for snp variants
indel_thr=0.96 #cnn output threshold for indel variants

N_snps=20 #number of SNP variants to take
N_indels=20 #number of INDEL variants to take

declare -A bams_matching

#BAM renaming
bams_matching=( ['PCAWG.3972cca7-91dc-4a27-9608-17c35cc4eb48.bam']=patient_1 ['PCAWG.66ae8aa3-0f01-44b2-9cae-5fde3a5bf31f.bam']=patient_2 \
['PCAWG.8524cdf9-fabc-4eb9-8561-670f7f5c2928.bam']=patient_3 ['PCAWG.af830047-822a-4dcc-88bc-68f7440a208d.bam']=patient_4 )

bam_filter=$(echo ${!bams_matching[@]}|sed 's/ /"||BAM="/g')
bam_filter='BAM="'$bam_filter'"' #choose only BAMs that are indices in bams_matching

gnomAD_filter='gnomAD_AF=0 || gnomAD_AF="."' #filter out all gnomAD variants

bcftools view -H -v snps -i "cnn_score>$snp_thr && SOMATIC=1 && ($bam_filter) && ($gnomAD_filter)" $output_vcf|grep -v snpEff_high|shuf|head -n $((N_snps/2))
bcftools view -H -v snps -i "cnn_score<$snp_thr && SOMATIC=0 && GERMLINE=1 && ($bam_filter) && ($gnomAD_filter)" $output_vcf|grep -v snpEff_high|shuf|head -n $((N_snps/4))
bcftools view -H -v snps -i "cnn_score<$snp_thr && SOMATIC=0 && GERMLINE=0 && ($bam_filter) && ($gnomAD_filter)" $output_vcf|grep -v snpEff_high|shuf|head -n $((N_snps/4))

bcftools view -H -v indels -i "cnn_score>$indel_thr && SOMATIC=1 && ($bam_filter) && ($gnomAD_filter)" $output_vcf|grep -v snpEff_high|shuf|head -n $((N_indels/2))
bcftools view -H -v indels -i "cnn_score<$indel_thr && SOMATIC=0 && GERMLINE=1 && ($bam_filter) && ($gnomAD_filter)" $output_vcf|grep -v snpEff_high|shuf|head -n $((N_indels/4))
bcftools view -H -v indels -i "cnn_score<$indel_thr && SOMATIC=0 && GERMLINE=0 && ($bam_filter) && ($gnomAD_filter)" $output_vcf|grep -v snpEff_high|shuf|head -n $((N_indels/4))
