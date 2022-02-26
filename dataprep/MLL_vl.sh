#!/bin/bash


project_name="MLL"
dataset_name="MLL_variants_list_20200804"

dataset_dir="/storage/groups/epigenereg01/workspace/projects/vale/datasets/snvs/${project_name}/${dataset_name}"

bam_dir="/storage/groups/epigenereg01/datasets/MLL-5000-genomes/matched_pairs/BAM/"

mll_vcf="/storage/groups/epigenereg01/workspace/projects/vale/mutation_tables/MLL/somatic/MLL_variants_list/MLL_variants_list_20200804.gnomAD.vcf.gz"

#bam_dir="/storage/groups/epigenereg01/workspace/projects/vale/data/icgc/${project_name}/bam/"

refgen_fa="/storage/groups/epigenereg01/workspace/projects/vale/calling/MLL/resources_GRCh37/GRCh37.fa"

negative_train_nn_vcf=$dataset_dir/vcfs/negative_train_nn.vcf.gz
negative_train_rf_vcf=$dataset_dir/vcfs/negative_train_rf.vcf.gz

negative_test_vcf=$dataset_dir/vcfs/negative_test.vcf.gz

somatic_train_vcf=$dataset_dir/vcfs/somatic_train.vcf.gz
somatic_test_vcf=$dataset_dir/vcfs/somatic_test.vcf.gz

images_dir=$dataset_dir/images_2

slurm_common_params="srun -p cpu_p --time=10:00:00 --nice=10000"
python_common_params="python generate_tensors.py --bam_dir $bam_dir --refgen_fa $refgen_fa --Lbatch 1 --tensor_width 150 --tensor_max_height 150 --tensor_crop_strategy topbottom --tensor_sort_by_variant 1"

#chroms=$(echo {1..22} X Y)

#for chrom in $chroms; do

#mkdir -p $images_dir/negative_train_nn/${chrom}
#$slurm_common_params -J negative_train_nn_${chrom}_${dataset_name} -o $images_dir/negative_train_nn/${chrom}/log.o -e $images_dir/negative_train_nn/${chrom}/log.e $python_common_params --output_dir $images_dir/negative_train_nn --chrom $chrom --vcf $negative_train_nn_vcf &

#mkdir -p $images_dir/negative_train_rf/${chrom}
#$slurm_common_params -J negative_train_rf_${chrom}_${dataset_name} -o $images_dir/negative_train_rf/${chrom}/log.o -e $images_dir/negative_train_rf/${chrom}/log.e $python_common_params --output_dir $images_dir/negative_train_rf --chrom $chrom --vcf $negative_train_rf_vcf &

#mkdir -p $images_dir/negative_test/${chrom}
#$slurm_common_params -J negative_test_${chrom}_${dataset_name} -o $images_dir/negative_test/${chrom}/log.o -e $images_dir/negative_test/${chrom}/log.e $python_common_params  --output_dir $images_dir/negative_test --chrom $chrom --vcf $negative_test_vcf &

mkdir -p $images_dir/somatic_test/
$slurm_common_params -J somatic_test_${dataset_name} -o $images_dir/somatic_test/log.o -e $images_dir/somatic_test/log.e $python_common_params  --output_dir $images_dir/somatic_test --vcf $mll_vcf  --tensor_check_variant_column 1 &

#mkdir -p $images_dir/somatic_train/${chrom}
#$slurm_common_params -J somatic_train_${chrom}_${dataset_name} -o $images_dir/somatic_train/${chrom}/log.o -e $images_dir/somatic_train/${chrom}/log.e $python_common_params --output_dir $images_dir/somatic_train --chrom $chrom --vcf $somatic_train_vcf  --image_check_variant_column 1 &

#for iter in $(seq 1 ${somatic_synth_iter}); do
#    mkdir -p $images_dir/somatic_synth_${iter}/${chrom}
#    $slurm_common_params -J somatic_synth_${iter}_${chrom}_${dataset_name} -o $images_dir/somatic_synth_${iter}/${chrom}/log.o -e $images_dir/somatic_synth_${iter}/${chrom}/log.e $python_common_params --output_dir $images_dir/somatic_synth_$iter --chrom $chrom --vcf $somatic_train_vcf  --image_check_variant_column 1 --replace_variant 1 --replace_variant_random_state $iter &
#done

#done
