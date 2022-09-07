#!/bin/bash

echo 'Choosing well-separable variants'

#./choose_variants.sh > best_vars

echo 'Generating per-sample BAM files and combined VCF'

#./generate_bams_and_vcf.sh best_vars|bcftools sort - -Oz -o BLCA-US_example_20220901.vcf.gz

#tabix -f BLCA-US_example_20220901.vcf.gz

echo 'Generating tensors'

source ~/.bashrc;conda activate dataprep

python ~/workspace/vale-variant-calling/paper/DeepSom/dataprep/generate_tensors.py \
  --tensor_sigma_noise=0.5 \
  --vcf=BLCA-US_example_20220901.vcf.gz --output_dir=$(pwd) --bam_dir=bam_files/ --refgen_fa="/lustre/groups/epigenereg01/workspace/projects/vale/calling/MLL/resources_GRCh37/GRCh37.fa"

rm variants.csv

mv BLCA-US_example_20220901_* BLCA-US_example_20220901.imgb

echo 'Making predictions'

source ~/.bashrc;conda activate nnc

python ~/workspace/vale-variant-calling/paper/DeepSom/cnn/nn.py --tensor_width=150 --tensor_height=70 --test_dataset=test_imgb.lst --model_weight=/lustre/groups/epigenereg01/workspace/projects/vale/nnc_logs/BLCA-US/train_test_flanking/gnomAD_thr_0/dropout_0.5/20220901/whole/weights/epoch_20_weights_model
