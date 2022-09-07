#!/bin/bash

#generate per-sample BAM files and the VCF file

variant_list=$1 #best_vars

BAM_dir='/lustre/groups/epigenereg01/workspace/projects/vale/data/icgc/BLCA-US/bam'

# unique_bams=($(cat best_vars|sed -E 's/.*BAM=([^\t^;]*).*/\1/'|sort|uniq))
#
# for i in ${!unique_bams[@]}; do
#   bam=${unique_bams[$i]}
#   bams_matching[$bam]=patient_$i
# done

rm -rf bam_files
mkdir bam_files

echo '##fileformat=VCFv4.2'
echo '##INFO=<ID=BAM,Number=.,Type=String,Description="BAM file name">'
#echo '##INFO=<ID=GERMLINE,Number=.,Type=Flag,Description="Germline variant">'
echo '##INFO=<ID=gnomAD_AF,Number=1,Type=Float,Description="gnomAD population allele frequency">'
echo '##INFO=<ID=SOMATIC,Number=.,Type=Flag,Description="Somatic variant">'
echo '##INFO=<ID=flanking,Number=.,Type=String,Description="Ref and alt AD of left and right flanking variants">'
echo '##contig=<ID=1,length=249250621>'
echo '##contig=<ID=2,length=243199373>'
echo '##contig=<ID=3,length=198022430>'
echo '##contig=<ID=4,length=191154276>'
echo '##contig=<ID=5,length=180915260>'
echo '##contig=<ID=6,length=171115067>'
echo '##contig=<ID=7,length=159138663>'
echo '##contig=<ID=8,length=146364022>'
echo '##contig=<ID=9,length=141213431>'
echo '##contig=<ID=10,length=135534747>'
echo '##contig=<ID=11,length=135006516>'
echo '##contig=<ID=12,length=133851895>'
echo '##contig=<ID=13,length=115169878>'
echo '##contig=<ID=14,length=107349540>'
echo '##contig=<ID=15,length=102531392>'
echo '##contig=<ID=16,length=90354753>'
echo '##contig=<ID=17,length=81195210>'
echo '##contig=<ID=18,length=78077248>'
echo '##contig=<ID=19,length=59128983>'
echo '##contig=<ID=20,length=63025520>'
echo '##contig=<ID=21,length=48129895>'
echo '##contig=<ID=22,length=51304566>'
echo '##contig=<ID=X,length=155270560>'
echo '##contig=<ID=Y,length=59373566>'
echo '##contig=<ID=MT,length=16569>'

printf '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n'

while read -r chrom pos id ref alt qual filter info; do
  bam=$(echo $info|sed -E 's/.*BAM=([^\t^;]*).*/\1/')
  new_bam=${bams_matching[$bam]}
  if [ ! -f bam_files/$new_bam.sam ];then samtools view -H  $BAM_dir/$bam|sed -e '/@PG/ d' -e 's/SM:[^ ]*/SM:'$new_bam'/'  -e 's/DT:[^T]*/DT:2022-09-01/' > bam_files/$new_bam.sam;fi
  samtools view $BAM_dir/$bam ${chrom}:$((pos-20))-$((pos+20)) >> bam_files/$new_bam.sam
  info="BAM=$new_bam.bam;"$(echo $info|sed -e 's/;/\n/g' | sed '/SOMATIC/b;/flanking/b;/gnomAD_AF/b;d'|tr '\n' ';'|sed 's/;$//')
  printf "$chrom\t$pos\t$id\t$ref\t$alt\t$qual\t$filter\t$info\n"
done <$variant_list

for bam in ${bams_matching[@]};do
  new_bam=bam_files/$bam
  samtools sort -O bam $new_bam.sam > $new_bam.bam
  samtools index $new_bam.bam
  #rm $new_bam.sam
done
