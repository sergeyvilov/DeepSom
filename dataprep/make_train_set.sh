#!/bin/bash

[ "$#" -lt 2 ] && echo "at least 2 arguments required, $# provided, exiting..." && exit

source ~/.bashrc; conda activate vale-bio

Project=$1 #project name
fold=$2

dataset_name="${3:-gnomAD_thr_0}"
max_gnomAD_AF="${4:-0}"  #gnomAD threshold

echo "Preparing dataset for $Project: $dataset_name with max_gnomAD_AF=$max_gnomAD_AF"

min_DP=1 #minimum coverage at the variant site

variants_per_split=1000

N_train_snps_negative=60000
N_train_indels_negative=15000 #x1.3 because many indels are filtered out during tensor generation
N_train_snps_somatic=60000
N_train_indels_somatic=10000

N_test_snps_negative=60000
N_test_indels_negative=15000 #x1.3 as because indels are filtered out during tensor generation
N_test_snps_somatic=60000
N_test_indels_somatic=35000

if [ "$Project" = "ESAD-UK" ] && [ "$dataset_name" =  "gnomAD_thr_1e-4" ];then
  N_train_snps_negative=100000
  N_train_indels_negative=52000 #x1.3 because many indels are filtered out during tensor generation
  N_train_snps_somatic=100000
  N_train_indels_somatic=40000
fi

extract_variants () {

  input_vcf_gz=$1
  output_tsv=$2
  N_variants_out=$3
  keep_germline_fraction="${4:-True}"     #keep the same fraction of germline variants as in the initial vcf

  if [[ $output_tsv =~ "somatic" ]];then
    #varclass_label=SOMATIC
    filtering_string="(gnomAD_AF='.'||gnomAD_AF<=$max_gnomAD_AF)&&(FORMAT/DP>$min_DP)&&(SOMATIC=1)"
    keep_germline_fraction=False
  else
    filtering_string="(gnomAD_AF='.'||gnomAD_AF<=$max_gnomAD_AF)&&(FORMAT/DP>$min_DP)&&(SOMATIC=0)"
    #varclass_label=NON-SOMATIC
  fi

  if [[ $output_tsv =~ "snps" ]];then
    variant_type=snps
    vartype_label=SNP
  else
    variant_type=indels
    vartype_label=INDEL
  fi

  #bcftools view -h $input_vcf_gz > $output_vcf

  prefiltered_vcf=prefiltered_vcf.temp.vcf

  bcftools view -H $input_vcf_gz -v "$variant_type" -i "$filtering_string" > $prefiltered_vcf

  if [[ $keep_germline_fraction == 'True' ]]; then

    echo "Preserving fraction of germline variants"
    N_germline_in=$(grep GERMLINE $prefiltered_vcf|wc -l) #number of germline variants in the initial vcf
    N_variants_in=$(wc -l $prefiltered_vcf|cut -d" " -f1) #total number of variants in the initial vcf

    germline_fraction=$(awk "BEGIN{print $N_germline_in/$N_variants_in}")
    echo "Fraction of germline variants in input vcf: $germline_fraction"

    N_germline_out=$(awk "BEGIN{print int($germline_fraction*$N_variants_out)}") #number of germline variants in the output vcf
    echo "Expected number of germline variants in the output vcf: $N_germline_out"

    N_artifacts_out=$(($N_variants_out-$N_germline_out))

    echo "Expected number of artefacts in the output vcf: $N_artifacts_out"

    grep GERMLINE $prefiltered_vcf|shuf -n $N_germline_out > $output_tsv
    grep -v GERMLINE $prefiltered_vcf|shuf -n $N_artifacts_out >> $output_tsv

  else

    shuf -n $N_variants_out $prefiltered_vcf > $output_tsv

  fi

  N_variants=$(wc -l $output_tsv|cut -d" " -f1) #actual number of output variants

  if [[ $output_tsv =~ "train" ]] && [ $N_variants -lt $N_variants_out ];then
    #we want a balanced train set
    echo "Insufficient number of train varints in output tsv ($N_variants instead of $N_variants_out), bootstrapping..."
    shuf -r -n $N_variants_out $output_tsv > $output_tsv.shuf
    mv $output_tsv.shuf $output_tsv

  fi

  #echo "Removing the FORMAT column and adding labels: $vartype_label;$varclass_label ..."
  #cat $output_tsv|awk -v varclass_label=$varclass_label -v vartype_label=$vartype_label 'BEGIN{OFS="\t"}{print $1,$2,$3,$4,$5,$6,$7,$8";"vartype_label";"varclass_label}' > $output_tsv.no_format_col
  #mv $output_tsv.no_format_col $output_tsv

}

vcfs_dir=/lustre/groups/epigenereg01/workspace/projects/vale/calling_new/$Project/all/results/5_fold/${fold}/ #source of non-somatic vcfs

output_dir=/lustre/groups/epigenereg01/workspace/projects/vale/datasets_new/nnc/${Project}/${dataset_name}/5_fold/${fold}/variants

if [ "$Project" != "TCGA-LAML" ]; then
  somatic_vcfs_dir="$vcfs_dir"
else
  somatic_vcfs_dir="/lustre/groups/epigenereg01/workspace/projects/vale/mutation_tables/TCGA-LAML/somatic/by_split/5_fold/${fold}"
fi


echo "output vcf dir: $output_dir"
mkdir -p $output_dir

cd $output_dir

printf "\n*Removing old subfolders\n"
rm -r test/ train/
printf "*Creating new subfolders\n"
mkdir -p test/ train/

printf "\n*Generating $N_train_snps_negative negative train snps\n"
extract_variants $vcfs_dir/train.vcf.gz negative_train_snps.temp.tsv $N_train_snps_negative
printf "\n*Generating $N_train_indels_negative negative train indels\n"
extract_variants $vcfs_dir/train.vcf.gz negative_train_indels.temp.tsv $N_train_indels_negative
printf "\n*Generating $N_train_snps_somatic somatic train snps\n"
extract_variants ${somatic_vcfs_dir}/train.vcf.gz somatic_train_snps.temp.tsv $N_train_snps_somatic
printf "\n*Generating $N_train_indels_somatic somatic train indels\n"
extract_variants ${somatic_vcfs_dir}/train.vcf.gz somatic_train_indels.temp.tsv $N_train_indels_somatic

printf "\n*Generating $N_test_snps_negative negative test snps\n"
extract_variants $vcfs_dir/test.vcf.gz negative_test_snps.temp.tsv $N_test_snps_negative
printf "\n*Generating $N_test_indels_negative negative test indels\n"
extract_variants $vcfs_dir/test.vcf.gz negative_test_indels.temp.tsv $N_test_indels_negative
printf "\n*Generating $N_test_snps_somatic somatic test snps\n"
extract_variants $somatic_vcfs_dir/test.vcf.gz  somatic_test_snps.temp.tsv $N_test_snps_somatic
printf "\n*Generating $N_test_indels_somatic somatic test indels\n"
extract_variants $somatic_vcfs_dir/test.vcf.gz  somatic_test_indels.temp.tsv $N_test_indels_somatic

printf "\n*Merging train tsv\n"
cat negative_train_snps.temp.tsv somatic_train_snps.temp.tsv|shuf > train_snps.tsv
cat negative_train_indels.temp.tsv somatic_train_indels.temp.tsv|shuf > train_indels.tsv

printf "\n*Merging test tsv\n"
cat negative_test_snps.temp.tsv somatic_test_snps.temp.tsv|shuf > test_snps.tsv
cat negative_test_indels.temp.tsv somatic_test_indels.temp.tsv|shuf > test_indels.tsv

printf "\n*Making train subtables\n"
cat train_snps.tsv|awk -v variants_per_split=$variants_per_split 'BEGIN{output_idx=0}{if (NR%variants_per_split==0){output_idx=NR}{print $0>"train/train_snps_"output_idx".tsv"}}'
cat train_indels.tsv|awk -v variants_per_split=$variants_per_split 'BEGIN{output_idx=0}{if (NR%variants_per_split==0){output_idx=NR}{print $0>"train/train_indels_"output_idx".tsv"}}'

printf "\n*Making test subtables\n"
cat test_snps.tsv|awk -v variants_per_split=$variants_per_split 'BEGIN{output_idx=0}{if (NR%variants_per_split==0){output_idx=NR}{print $0>"test/test_snps_"output_idx".tsv"}}'
cat test_indels.tsv|awk -v variants_per_split=$variants_per_split 'BEGIN{output_idx=0}{if (NR%variants_per_split==0){output_idx=NR}{print $0>"test/test_indels_"output_idx".tsv"}}'

bcftools view -H $vcfs_dir/snpEff_high.test.vcf.gz|sed 's/ANN=[^;^\t]*/ANN=snpEff_high/'|awk -v variants_per_split=$variants_per_split 'BEGIN{output_idx=0}{if (NR%variants_per_split==0){output_idx=NR}{print $0>"test/test_snpEff_high_"output_idx".tsv"}}'

rm *temp*

echo "Done"
