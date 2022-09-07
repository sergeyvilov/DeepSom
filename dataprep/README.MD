# DeepSom data preprocessing

DeepSom can be used in train, evaluation (test) or inference mode. In all the three modes, the [CNN](../CNN) accepts SNP/INDEL variants encoded in the form of 3D variant tensors. Here, we describe how variant tensors should be prepared.

## Setup dependencies

Install  [miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

Then run:

```
conda env create -f environment.yml
conda activate DeepSom-dataprep
```

We shall first describe how to prepare tensors for inference since it involves the smallest number of steps.

## Inference mode

1. Call all possible variants (somatic, germline + artefacts)

Using [Mutect2](https://gatk.broadinstitute.org/hc/en-us/articles/360037593851-Mutect2) in tumour-only mode we call all possible
variants based on the sample BAM file.

Right after calling, do NOT merge VCF files for all WGS samples. Each VCF file must contain variants for one sample only.

For the next steps, keep all BAM files in a single folder.

In the following, we will discuss tensor preparation for one of these VCF files, let it be `wgs_sample_name.vcf.gz`.

2. Annotate for gnomAD population allele frequency

This step can be skipped if gnomAD-based variant pre-filtering is not needed and the flanking regions feature is
not going to be used for classification.

For each called variant, we add information about the population allele frequency (AF) in the [gnomAD](https://gnomad.broadinstitute.org/) database. A single VCF file can be annotated as follows:

```
annotation_file=gnomAD_dir/gnomAD_light.vcf.gz #path to lightweight gnomAD VCF

echo '##INFO=<ID=gnomAD_AF,Number=1,Type=Float,Description="Alternative allele frequency as in GNOMAD v.3.1.1">' > gnomad_header.txt

tabix wgs_sample_name.vcf.gz

bcftools annotate --threads 4  \
-c 'ID,INFO/gnomAD_AF:=INFO/AF' \
-h gnomad_header.txt \
-a $annotation_file \
wgs_sample_name.vcf.gz \
-Oz -o wgs_sample_name.gnomAD.vcf.gz
```

The gnomAD database consists of per-chromosome VCF files and is quite huge. Annotating with it would take many hours.
However, we only need to know AF of each gnomAD variant. So, we can make a single VCF file `gnomAD_light.vcf.gz` by concatenating all per-chromosome VCF files and discarding all information in the INFO field except AF. Annotating with this lightweight version of gnomAD takes less than 30 minutes per WGS sample.

Watch out for the gnomAD reference genome: it should be the same as the one used in Mutect2 calling, otherwise [liftover](https://genome.sph.umich.edu/wiki/LiftOver) on gnomAD variants is needed.

3. Annotate for flanking regions

This step makes sense only for gnomAD-annotated VCFs. At this step we include information about
*likely germline* variants adjacent to each candidate variant. This should help take local variations of germline variant allele fraction (VAF) into account.

The annotation command is as follows:

```
min_germline_gnomAD_AF=0.1 #minimal gnomAD AF to consider the variant germline
flanking_region_length=2000000 #flanking region left and right length

python get_flanking.py wgs_sample_name.gnomAD.vcf.gz $min_germline_gnomAD_AF $flanking_region_length > wgs_sample_name.flanking

flanking_params="(flanking_variants=2+2, min_germline_gnomAD_AF={params.min_germline_gnomAD_AF}, flanking_region_length={params.flanking_region_length})"

bcftools view -h wgs_sample_name.gnomAD.vcf.gz \
| sed -E '10i ##INFO=<ID=flanking,Number=.,Type=String,Description=\"Flanking regions: left VAF|left DP|right VAF|right DP '"$flanking_params"'\">' > wgs_sample_name.flanking.vcf

bcftools view -H wgs_sample_name.gnomAD.vcf.gz | paste - wgs_sample_name.flanking | awk 'BEGIN {OFS="\t"} {$8=$8";flanking="$NF; $NF=""; print $0}' >> wgs_sample_name.flanking.vcf

bgzip wgs_sample_name.flanking.vcf
```

4. Remove all variants with gnomAD AF above the threshold.

```
gnomAD_AF_thr=0 #maximal gnomAD AF

bcftools view -i 'gnomAD_AF<'$gnomAD_AF_thr' || gnomAD_AF="."' wgs_sample_name.flanking.vcf.gz -Oz -o wgs_sample_name.filtered.vcf.gz
```

5. Add BAM information to the INFO field

The `generate_tensors.py` function should be able to access raw reads corresponding
to each called variant. So, each variant in the VCF file should be annotated with the
name of the corresponding BAM file:

```
bam_file_name=sample_XYZ.bam #BAM file corresponding to this VCF file, don't include the full path

bcftools view -h wgs_sample_name.filtered.vcf.gz \
| sed -E '10i ##INFO=<ID=BAM,Number=.,Type=String,Description=\"BAM sample name\">' > wgs_sample_name.bam.vcf

bcftools view -H wgs_sample_name.filtered.vcf.gz | awk -v bam_file_name=$bam_file_name 'BEGIN {OFS="\t"} {$8=$8";BAM="bam_file_name; print $0}' >> wgs_sample_name.bam.vcf

bgzip wgs_sample_name.bam.vcf
```

Use only the BAM basename, without the full path.

As soon as the BAM name is added to the INFO field, VCF files from several samples can be concatenated. For concatenation to be successfull, all concatenated VCF files must have the same sample name in the header. To change the sample name in the header, use the following commands:

```
bcftools view wgs_sample_name.bam.vcf.gz | sed '/#CHROM/ s/FORMAT\t.*/FORMAT\tSAMPLE/' | bgzip -c > wgs_sample_name.unified_sample.vcf.gz

tabix wgs_sample_name.unified_sample.vcf.gz
```

Alternatively, FORMAT and SAMPLE columns can simply be removed since they are ignore during tensor generation.

To concatenate several VCF files use the following command:

```
bcftools concat -a wgs_sample_name_1.same_sample.vcf.gz wgs_sample_name_2.same_sample.vcf.gz -Oz -o inference.vcf.gz
```

See [BLCA-US_example_20220901.vcf](../test/BLCA-US/BLCA-US_example_20220901.vcf) to learn how the final VCF file should look like  (FORMAT and SAMPLE columns are removed).

6. Generate imgb batches

Finally, the VCF file can be used to generate variant tensors:

```
python generate_tensors.py \
--vcf './inference/project_name/wgs_sample_name/vcf/wgs_sample_name.unified_sample.vcf.gz' \
--output_dir './inference/project_name/wgs_sample_name/tensors/' \
--bam_dir './bam/project_name/' \
--refgen_fa './ref/GRCh37.fa' \
--tensor_width 150 \
--tensor_max_height 70
```

`bam_dir` --- folder with BAM files for all variants in the `--vcf`

`refgen_fa` --- reference genome (genome used for variant calling with Mutect2) FASTA file

`--tensor_width` --- width of the variant tensor

`--tensor_height` --- maximal height of the variant tensor. Variant tensors with more reads will be cropped.

The VCF header is actually ignored by `generate_tensors.py`, so it does not need to be preserved.

To speed up I/O operations, `generate_tensors.py` combines variant tensors into large imgb batches, 5K variants each.

7. When imgb batches are generated, descend to the `output_dir` and make a list of all batches:

```
find ~+  -name '*.imgb' > wgs_sample_name_imgb.lst
```
This list is to be passed to the CNN using the `--test_dataset` option.

## Test(evaluation) mode

Data preparation in test mode includes all steps of data preparation in inference mode.

However, test (and train) modes are only possible when a matched normal is available for a tumor sample, s.t.
somatic variants can be identified. Somatic variants can be called, for example, by additionally running Mutect2 on a tumor-normal pair.

Somatic variants should be labelled with tag SOMATIC in the INFO field of the VCF file. Non-somatic variants
do not need to be labelled. See [BLCA-US_example_20220901.vcf](../test/BLCA-US/BLCA-US_example_20220901.vcf) to learn how the tag SOMATIC should be added.

Note that WGS samples used in evaluation must not be used in CNN training.

## Train mode

For training the CNN one needs around 120 000 SNPs (50% somatic/50% non-somatic) and
20 000 INDELs (50% somatic/50% non-somatic).

These variants should be collected from several WGS
samples, s.t. the full range of read depths, mutational signatures, etc. is represented.

When the number of somatic variants is below the mentioned values, upsampling should be done.

So, to prepare train data, one needs the following:

1. Choose several WGS samples with known somatic variants (or call somatic variants by applying Mutect2 on tumor-normal pairs). Follow all steps for test data preparation described above.

2. Combine per-sample VCF file into a single VCF file `train.vcf.gz` using the `bcftools concat -a` command. Note
that for a successfull concatenation the VCF files should have the same sample name in the header.

3. Randomly choose the required number of somatic and non-somatic SNP and INDEL variants from `train.vcf.gz`.
Upsample when necessary. See example commands in `make_train_set.sh`, which generates `train.selected.tsv` required
in the next step.

4. Generate imgb tensor batches:

```
python generate_tensors.py \
--vcf './train/train_dataset_name/vcf/train.selected.tsv' \
--output_dir './train/train_dataset_name/tensors/' \
--bam_dir './bam/project_name/' \
--refgen_fa './ref/GRCh37.fa' \
--tensor_width 150 \
--tensor_max_height 70
```

When using a computational cluster, one can speed up data preparation by splitting `train_selected.vcf` into many
small files and running a number of parallel instances of `generate_tensors.py`.

5. When imgb batches are generated, descend to the `output_dir` and make a list of all batches:

```
find ~+  -name '*.imgb' > train_imgb.lst
```
This list is to be passed to the CNN using the `--train_dataset` option.
