# Data preparation for the Neural Network Classifier (NNC)

The NNC can be used in train, evaluation (test) or inference mode. In all the three modes, the NNC accepts SNP/INDEL variants encoded in the form of 3D variant tensors. Here, we describe how to prepare such variant tensors.

The dependencies for the data preparation step can be installed as described below.

## Setup

Install  [miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

Then run:

```
conda env create -f environment.yml
conda activate dataprep
```

##Data preprocessing in inference mode

We shall first describe how to prepare tensors for inference on a single WGS sample.
In inference mode, we don't know true variant labels (somatic/non-somatic) and use the NNC to classify the variants.

Data preparation consists of several steps:

1. Call all possible variants (somatic, germline + artefacts)

Using [Mutect2](https://gatk.broadinstitute.org/hc/en-us/articles/360037593851-Mutect2) in tumour-only mode we call all possible
variants in the sample.

For the next steps, do NOT merge VCF files for all samples. Each VCF file must contain variants for one sample only.

In the following, we will discuss tensor preparation for one VCF file, let it be `wgs_sample_name.vcf.gz`.

2. Annotate for gnomAD population allele frequency

This step can be skipped if variant pre-filtering is not required and the flanking regions feature is
not going to be used for classification.

For each called variant, we add information about the population allele frequency in the gnomAD database. A single VCF file can be annotated using the following command:

```
annotation_file=gnomAD_dir/gnomAD_light.vcf.gz #path to lightweight gnomAD VCF

tabix wgs_sample_name.vcf.gz

echo '##INFO=<ID=gnomAD_AF,Number=1,Type=Float,Description="Alternative allele frequency as in GNOMAD v.3.1.1">' > gnomad_header.txt

bcftools annotate --threads 4  \
-c 'ID,INFO/gnomAD_AF:=INFO/AF' \
-h gnomad_header.txt \
-a $annotation_file \
wgs_sample_name.vcf.gz \
-Oz -o wgs_sample_name.gnomAD.vcf.gz
```

The gnomAD database consists of per-chromosome VCF files and is quite huge. Annotating with it would take many hours.
However, we need only the allele frequency (AF) information of each gnomAD variant. So, we make a single VCF file `gnomAD_light.vcf.gz` by concatenating all per-chromosome VCF files and discarding all information in the INFO field except AF. Annotating with this lightweight version of the gnomAD database takes less than 30 minutes per WGS sample.

Watch out for the gnomAD reference genome: it should be the same as one used for Mutect2 calling, otherwise liftover on gnomAD variants is needed.

3. Annotate for flanking regions

This step makes sense only if gnomAD annotation was done. The idea is to include information about
adjacent likely germline variants to take local variations of germline variant allele fraction (VAF) into account.

The annotation command is as follows:

```
min_germline_gnomAD_AF=0.1 #minimal gnomAD AF to consider the variant germline
flanking_region_length=2000000 #flanking region left and right length

python get_flanking.py wgs_sample_name.gnomAD.vcf.gz $min_germline_gnomAD_AF $flanking_region_length > wgs_sample_name.flanking

flanking_params="(flanking_variants=2+2, min_germline_gnomAD_AF={params.min_germline_gnomAD_AF}, flanking_region_length={params.flanking_region_length})"

bcftools view -h wgs_sample_name.gnomAD.vcf.gz \
|sed -E '10i ##INFO=<ID=flanking,Number=.,Type=String,Description=\"Flanking regions: left VAF|left DP|right VAF|right DP '"$flanking_params"'\">' > wgs_sample_name.flanking.vcf

bcftools view -H wgs_sample_name.gnomAD.vcf.gz |paste - wgs_sample_name.flanking | awk 'BEGIN {OFS="\t"} {$8=$8";flanking="$NF; $NF=""; print $0}' >> wgs_sample_name.flanking.vcf

bgzip wgs_sample_name.flanking.vcf
```

4. Remove all variants with gnomAD AF above the threshold.

```
gnomAD_AF_thr=1e-4 #maximal gnomAD AF

bcftools view -i 'gnomAD_AF<'$gnomAD_AF_thr' || gnomAD_AF="."' wgs_sample_name.flanking.vcf.gz -Oz -o wgs_sample_name.filtered.vcf.gz
```

5. Add BAM information to the INFO field

The `generate_tensors.py` function should be able to access raw reads corresponding
to each of the called variants. So, each variant in the VCF file should be annotated with the
name of the corresponding BAM file:

```
bam_file_name=sample_XYZ.bam #BAM file corresponding to this VCF file, use only basename

bcftools view -h wgs_sample_name.filtered.vcf.gz \
|sed -E '10i ##INFO=<ID=BAM,Number=.,Type=String,Description=\"BAM sample name\">' > wgs_sample_name.bam.vcf

bcftools view -H wgs_sample_name.filtered.vcf.gz |awk -v bam_file_name=$bam_file_name 'BEGIN {OFS="\t"} {$8=$8";BAM="bam_file_name; print $0}' >> wgs_sample_name.bam.vcf

bgzip wgs_sample_name.bam.vcf
```

Use only the BAM basename, without the full path.

As soon as the BAM name is added to the INFO field, VCF files from several samples can be concatenated. All concatenated VCF files must have the same sample name in the header. To change the sample name in the header, use the following commands:

```
bcftools view wgs_sample_name.bam.vcf.gz |sed '/#CHROM/ s/FORMAT\t.*/FORMAT\tSAMPLE/'| bgzip -c > wgs_sample_name.unified_sample.vcf.gz

tabix wgs_sample_name.unified_sample.vcf.gz
```

See `example.unified_sample.vcf` to learn how a single-sample VCF file should look like after the previous steps.

To concatenate several VCF files use the following command:

```
bcftools concat -a wgs_sample_name_1.same_sample.vcf.gz wgs_sample_name_2.same_sample.vcf.gz -Oz -o inference.vcf.gz
```

BAM files of all concatenated samples should be in a single folder.

6. Generate imgb tensor batches

Finally, the VCF file can be used to generate variant tensors. To speed up inference, we can group variant tensors in .imgb batches, each including 512 variants:

```
python generate_tensors.py \
--vcf './inference/project_name/wgs_sample_name/vcf/wgs_sample_name.unified_sample.vcf.gz' \
--output_dir './inference/project_name/wgs_sample_name/tensors/' \
--bam_dir './bam/project_name/' \
--refgen_fa './ref/GRCh37.fa' \
--Lbatch 512 \
--tensor_width 150 \
--tensor_max_height 70
```

`bam_dir` --- folder with BAM files for all variants in the `--vcf`

`refgen_fa` --- reference genome (genome used for variant calling with Mutect2) FASTA file

`Lbatch` --- how many variants should be included in each imgb batch

`--tensor_width` --- width of the variant tensor

`--tensor_height` --- maximal height of the variant tensor. Variant tensors with more reads will be cropped.

7. When imgb batches are generated, descend to the `output_dir` and make a list of all batches:

```
find ~+  -name '*.imgb' > wgs_sample_name_imgb.lst
```
This list is to be passed to the NNC using the `--test_dataset` option.

##Data preprocessing in test(evaluation) mode

In test mode, we use variants with known labels (somatic/non-somatic) to evaluate the NNC performance.

Somatic variants should be labelled with tag SOMATIC in the INFO field of the VCF file. Non-somatic variants
do not need to be labelled.

Data preparation in test mode includes all steps of data preparation in inference mode.

However, test (and train) modes are only possible when a matched normal is available for a tumor sample, s.t.
somatic variants can be identified.

To call somatic variants and label them in the VCF file, run Mutect2 again using a matched normal.

Note that WGS samples used for evaluation must not be used for training.

##Data preprocessing in train mode.

For training the NNC one needs around 120 000 SNPs (50% somatic,50% non-somatic) and
20 000 INDELs (50% somatic,50% non-somatic).

These variants should be collected from several WGS
samples, s.t. the full range of read depths, mutational signatures, etc. is represented.

When the number of somatic variants is below the mentioned values, upsampling is required.

So, to prepare train data, we need the following:

1. Choose several WGS samples for training. Follow all steps for test data preparation described in the previous section.

2. Combine per-sample VCF file into a single VCF file `train.vcf.gz` using the `bcftools concat -a` command. Note
that for a successfull concatenation the VCF files should have the same sample name.

3. Randomly choose the required number of somatic and non-somatic SNP and INDEL variants from `train.vcf.gz`.
Upsample when necessary. See example commands in the script `make_train_set.sh`.

4. Generate imgb tensor batches:

```
python generate_tensors.py \
--vcf './train/train_dataset_name/vcf/train_variants.vcf.gz' \
--output_dir './train/train_dataset_name/tensors/' \
--bam_dir './bam/project_name/' \
--refgen_fa './ref/GRCh37.fa' \
--Lbatch 32 \
--tensor_width 150 \
--tensor_max_height 70
```

Note that `Lbatch=32` is important for correct model training, it is not recommended to change this value.

When using a computational cluster, one can speed up data preparation by splitting `train_selected.vcf` into many
small VCF files and running a number of parallel instances of `generate_tensors.py`.

5. When imgb batches are generated, descend to the `output_dir` and make a list of all batches:

```
find ~+  -name '*.imgb' > train_imgb.lst
```
This list is to be passed to the NNC using the `--train_dataset` option.
