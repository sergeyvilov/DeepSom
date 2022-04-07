#!/bin/bash

######
#Example script for generating variant tensors based on SNP variants
#for neural network training and performance evaluation
#
#Input parameters:
#negative_vcf -  .vcf.gz file with germline variants and sequencing artefacts, output of Mutect2 in tumor-only mode
#somatic_vcf -  .vcf.gz file with somatic variants, output of a somatic variant calling pipeline
#bam_dir - directory with BAM files of all samples in negative_vcf and somatic_vcf
#bam_matching_csv - matching table between BAM sample name and BAM file name
#output_dir - output dir for tensors and train/test VCFs
#refgen_fa - reference genome .fa file
#
#negative_vcf and somatic_vcf should contain a gnomad_AF record in the INFO field with gnomAD pupolation AF frequency
######


####
#dataset generation parameters
max_gnomAD_AF=0.0 #filter out variants with gnomAD AF below this value
N_train=20000 #total number of train variants (somatic+negative)
N_test=10000 #total number of test variants (somatic+negative)

#tensor generation parameters
Lbatch=4 #number of tensors in each imgb batch
tensor_width=150 # tensor width
tensor_max_height=75 #max tensor height
tensor_crop_strategy='topbottom' #how to crop variant tensor when read depth>tensor_height
tensor_sort_by_variant=1 #sort reads by base in the variant column
tensor_check_variant='snps' #check if the variant is a valid SNP and the alternative allele is present in actual pileup
####

negative_vcf='./projects/project_name/calling/negative/results/negative.vcf.gz' #VCF file with germline variants and artefacts, output of Mutect2

somatic_vcf='./projects/project_name/calling/somatic/results/somatic.vcf.gz' #VCF file with somatic variants, output of a somatic variant calling pipeline

output_dir='./datasets/dataset_name/' #dir for train/test VCF files and tensors

bam_dir='./projects/project_name/BAMs' #directory with BAM files

bam_matching_csv=$bam_dir/bam_matching.csv #matching between BAM sample name in BAM file name: each line should have comma-separated BAM sample name and BAM file name (without the path)

refgen_fa='./ref/GRCh37.fa' #reference genome FASTA file


conda activate dataprep
####

echo "Preparing train and test VCF files"

# Perform gnomAD filtering on input VCF files according to max_gnomAD_AF and choose only bialleic SNPs
# Split VCFs in train and test sets according to N_train and N_test

mkdir -p $output_dir/vcfs

#VCF files to create
negative_train_vcf=$output_dir/vcfs/negative_train.vcf.gz
negative_test_vcf=$output_dir/vcfs/negative_test.vcf.gz

somatic_train_vcf=$output_dir/vcfs/somatic_train.vcf.gz
somatic_test_vcf=$output_dir/vcfs/somatic_test.vcf.gz

echo "Removing somatic variants from negative_vcf, applying the gnomAD AF cutoff, selecting bialleic SNP variants"
negative_filtered_vcf=$output_dir/negative_filtered.vcf.gz
bcftools isec -C -w1 -i "(gnomAD_AF='.'||gnomAD_AF<=$max_gnomAD_AF)" $negative_vcf $somatic_vcf|bcftools view -v snps --max-alleles 2 -|bgzip -c >  $negative_filtered_vcf
tabix -f $negative_filtered_vcf

echo "Generating negative train vcf"
bcftools view -h $negative_filtered_vcf|bgzip -c > $negative_train_vcf #header only
#select SNPs only and choose randomly $N_train/2 variants
bcftools view -H $negative_filtered_vcf|shuf -n $(($N_train/2))|sort -k1,1 -k2,2n|bgzip -c >> $negative_train_vcf
tabix -f $negative_train_vcf

echo "Generating negative test vcf"
bcftools view -h $negative_filtered_vcf|bgzip -c > $negative_test_vcf #header only
#select SNPs that were not chosen for training and choose randomly $N_test/2 variants
bcftools isec -C -w1 $negative_filtered_vcf $negative_train_vcf|bcftools view -H -|shuf -n $(($N_test/2))|sort -k1,1 -k2,2n|bgzip -c >> $negative_test_vcf
tabix -f $negative_test_vcf

rm $negative_filtered_vcf
rm $negative_filtered_vcf.tbi

echo "Applying the gnomAD AF cutoff and selecting bialleic SNP variants in somatic_vcf"
somatic_filtered_vcf=$output_dir/somatic_filtered.vcf.gz
bcftools view -i "(gnomAD_AF='.'||gnomAD_AF<=$max_gnomAD_AF)" $somatic_vcf|bcftools view -v snps --max-alleles 2 -|bgzip -c > $somatic_filtered_vcf
tabix -f $somatic_filtered_vcf

echo "Generating somatic train vcf"
bcftools view -h $somatic_filtered_vcf|bgzip -c > $somatic_train_vcf #header only
#select SNPs only and choose randomly $N_train/2 variants
bcftools view -H $somatic_filtered_vcf|shuf -n $(($N_train/2))|sort -k1,1 -k2,2n|bgzip -c >> $somatic_train_vcf
tabix -f $somatic_train_vcf

echo "Generating somatic test vcf"
bcftools view -h $somatic_filtered_vcf|bgzip -c > $somatic_test_vcf #header only
#select SNPs that were not chosen for training and choose randomly $N_test/2 variants
bcftools isec -C -w1 $somatic_filtered_vcf $somatic_train_vcf|bcftools view -H -|shuf -n $(($N_test/2))|sort -k1,1 -k2,2n|bgzip -c >> $somatic_test_vcf
tabix -f $somatic_test_vcf

rm $somatic_filtered_vcf
rm $somatic_filtered_vcf.tbi
####

echo "Generating variant tensors"

python_common_params="python generate_tensors.py \
--bam_dir $bam_dir \
--bam_matching_csv $bam_matching_csv \
--refgen_fa $refgen_fa \
--Lbatch $Lbatch \
--tensor_width $tensor_width \
--tensor_max_height $tensor_max_height \
--tensor_crop_strategy $tensor_crop_strategy \
--tensor_sort_by_variant $tensor_sort_by_variant \
--tensor_check_variant $tensor_check_variant_column" #parameters of images generation

images_dir=$output_dir/tensors

#generate tensors for all chromosomes in parallel

chroms=$(echo {1..22} X Y)

for chrom in $chroms; do

  mkdir -p $images_dir/negative_train/${chrom}
  $python_common_params --output_dir $images_dir/negative_train --chrom $chrom --vcf $negative_train_vcf > $images_dir/negative_train/${chrom}/log 2>&1 &

  mkdir -p $images_dir/negative_test/${chrom}
  $python_common_params  --output_dir $images_dir/negative_test --chrom $chrom --vcf $negative_test_vcf > $images_dir/negative_test/${chrom}/log 2>&1 &

  mkdir -p $images_dir/somatic_test/${chrom}
  $python_common_params  --output_dir $images_dir/somatic_test --chrom $chrom --vcf $somatic_test_vcf > $images_dir/somatic_test/${chrom}/log 2>&1 &

  mkdir -p $images_dir/somatic_train/${chrom}
  $python_common_params --output_dir $images_dir/somatic_train --chrom $chrom --vcf $somatic_train_vcf > $images_dir/somatic_train/${chrom}/log 2>&1 &

done
