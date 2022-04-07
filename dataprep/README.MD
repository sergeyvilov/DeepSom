# Data preparation for the Neural Network Classifier (NNC)

The NNC can be used in train, evaluation (test) or inference mode. In all the three modes, the NNC accepts SNP/INDEL variants encoded in the form of 3D variant tensors. To obtain these tensors, one should first prepare a VCF file with corresponding variants.

## Setup

Install  [miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

Then run:

```
conda env create -f environment.yml
conda activate dataprep
```

## Preparing VCF files

1. Train mode

Train tensors can be generated based on two VCF files: one for the positive (somatic) class and one for the negative (germline variants plus artefacts) class. Note that variants used for training can not be used later for evaluation.

The somatic VCF can be obtained using standard somatic variant calling pipelines for a number of tumour samples for which a matched control is available. The negative VCF can be obtained by running a somatic variant caller (e.g. [Mutect2](https://gatk.broadinstitute.org/hc/en-us/articles/360037593851-Mutect2)) on these samples in tumour-only mode. The true somatic variants should then be removed from the negative VCF. This can be done, for example, with the following command:

```
bcftools isec -C -w1 negative.vcf.gz somatic.vcf.gz -Oz -o negative_clean.vcf.gz
```

From both negative and somatic VCFs, all variants that are not bialleic should be removed:

```
bcftools view --max-alleles 2 negative_clean.vcf.gz -Oz -o negative_clean_filtered.vcf.gz
```

We recommend that each of the VCFs has at least 10 000 records for SNP variants and 10 000 records for INDEL variants. More records will, in general, provide a better performance.

To achieve class balance at training, the somatic VCF should have approximately the same number of records as the negative VCF.
If less somatic variants are available, the option `--resample` should be used when training the NNC.

The NNC will generalize better when the somatic VCF accumulates variants from several different tumour samples. It is also highly recommended that the negative VCF combines germline variants and artefacts from different samples.

Note that any preprocessing (e.g. gnomAD filtering) should be applied before the desired number of variants are selected. This preprocessing should be applied in exactly the same way to the somatic VCF and the negative VCF.

2. Evaluation (test) mode

The VCFs can be obtained in the same way as for the train mode. The number of variants for evaluation is usually lower: about 20% of the train set. More variants will lead to a smaller variance in NNC performance estimation. Note that generating test VCFs and variant tensors is not always necessary: when training the NNC, one can allocate a given proportion of the train set for evaluation using the `--val_fraction` option.

3. Inference mode

The true labels are unknown, only one VCF is needed.

Note that all required preprocessing (e.g. gnomAD filtering) should be applied exactly in the same way when preparing train/test or inference VCFs.


## Generating variant tensors

Variant tensors can be generated from a VCF file with the following command:

```
python generate_tensors.py \
--vcf './datasets/dataset_name/vcfs/test.vcf.gz' \
--output_dir './datasets/dataset_name/tensors/test' \
--bam_dir './projects/project_name/BAMs' \
--bam_matching_csv './projects/project_name/BAMs/matching.csv' \
--refgen_fa './ref/GRCh37.fa' \
--Lbatch 4 \
--tensor_width 150 \
--tensor_max_height 70
```

* A variant tensor is made of reads collected for a given record in a VCF file. So, for each VCF record, the script `generate_tensors.py` should know which BAM file to use. The name of the BAM file can be either directly encoded in the INFO field of the VCF record (under a `BAM=file_name_without_path.bam` record) or provided in a .csv table which is passed with `--bam_matching_csv` parameter to the script `generate_tensors.py`. Each row of this table should have comma-separated BAM sample name and the corresponding BAM file name (without the path).

* VCFs used for tensors generation must only include bialleic mutations.

* The parameters `--tensor_width` and `--tensor_max_height` define the width and the maximal height of the variant tensor correspondingly. High values of these parameters may lead to a better performance but a larger  training time. The parameter `--tensor_max_height` is usually chosen s.t. the probability of having a variant with a larger read depth is small (e.g. the 75th percentile of the read depth distribution). The maximal reasonable value for `--tensor_width` is twice the most probable read length, but we note that tuning `--tensor_width` does not change the NNC performance much: the observed ROC AUC score for the TCGA-LAML dataset was only by about 0.5\% lower with `--tensor_width=75` compared to `--tensor_width=150` (twice the most probable read length). The minimal value of `--tensor_width` is 24 (limited by the NNC architecture).

* Set parameter `--tensor_check_variant` to filter out invalid variants during tensor generation:
 `snps`  to filter out variants that are not valid SNPs
 `indels`  to filter out variants that are not valid INDELs
 `vaf_only` to filter out variants when the alternative allele doesn't appear in the pileup (happens with Mutect2 output)

* To speed up training, we group variant tensors in .imgb batches of 4. When creating a train set, this is the recommended value for `--Lbatch`  as other values may affect the recommended NNC hyperparameters. When generating tensors for evaluation/inference, any positive integer value of `--Lbatch` can be used. Larger values lead to faster inference.

* Inference .imgb batches should be generated in a way that tracing of original variants is possible. So,
each row in the VCF file should correspond to only one sample. When more samples share a variant,
separate VCF rows should be created for each of the samples (each row having its own BAM= record). Alternatively,
a separate VCF file can be created for each sample.

* The script `generate_tensors.py` creates .imgb batches in the output folder, each batch comprising `--Lbatch` variant tensors. Each batch is named using the index of the VCF record corresponding to the first variant in the batch. Batches are distributed over subfolders which are automatically created in the output folder, at most 100 batches per subfolder.

* To speed up tensors generation, one can launch `generate_tensors.py` for multiple chromosomes in parallel using the `--chrom` parameter. Use `--chrom_start` and `--chrom_stop` to choose a particular region within the chromosome.

* See `python generate_tensors.py --help` for more training options.

* After generating the tensors, the information about variants and corresponding .imgb batches can be found in a `variants.csv.gz`
file in the `output_dir`.

* Each *.imgb batch is a python pickle object with a dictionary of two items: "images" and "info". "images" is a list of variant tensors of length `Lbatch`. "info" is a list of tensor descriptions of length `Lbatch`. Each item in "info" corresponds to one tensor in "images" and includes information
about the BAM file, ref and alt alleles, position, chromosome, etc. of the variant.

After generating .imgb batches, descend to the `output_dir` and run the following command to get a list of all tensor batches:

```
find ~+  -name '*.imgb' > all_imgb.lst
```

The example script `make_tensors.sh` demonstrates how to perform gnomAD filtering, choose bialleic variants, make train/test split and create variant tensors based on VCFs for SNP variants. INDEL variants can be processed in a similar way.
