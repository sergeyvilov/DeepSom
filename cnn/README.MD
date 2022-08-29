# DeepSom Convolutional Neural Network (CNN) Classifier.

## Setup

Install  [miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

Then run:

```
conda env create -f environment.yml
conda activate DeepSom-cnn
```

## Inference mode.

This is the main operating mode. In inference mode, the CNN assigns a pseudoprobability score to each candidate variant. This score can then be used for variant classification.

An example CNN run in inference mode:

```
python nn.py \
--test_dataset './inference/project_name/wgs_sample_name/tensors/wgs_sample_name_imgb.lst' \
--output_dir './inference/project_name/wgs_sample_name/results/' \
--model_weight './train/train_dataset_name/results/weights/epoch_N_weights_model' \
--tensor_width 150 \
--tensor_height 70 \
--max_depth 150 \
```

`--model_weight` --- path to the weights of a pretrained model.

`--tensor_width` --- width of the variant tensor

`--tensor_height` --- maximal height of the variant tensor. The VAF resolution of 1/`tensor_height` is guaranteed *only if the same height was used for tensor generation*, otherwise tensors with a larger height are just cropped.

`--max_depth` --- maximal read depth (used to normalize the flanking regions feature). Since the read depth distribution
often has a long right tail, we recommend using the 99th quantile of this distribution as `--max_depth` (about 2x median coverage)

Predictions on `test_dataset` are saved to `output_dir/predictions/final_predictions.vcf`, where the CNN score is stored under `cnn_score` record in the INFO field.

`--tensor_width`, `--tensor_height` and `--max_depth` **should correspond to those used in train mode**.

## Test(evaluation) mode

This mode is used for DeepSom evaluation. Evaluation can be performed only on variants with known labels (somatic/non-somatic).
To run the CNN, use the same command as in inference mode (see above).

## Train mode

In train mode, the CNN learns relationships between variant tensors and variant labels (somatic/non-somatic). The goal of the training step is to generate a CNN model (distribution of weights) which can then be used for inference on previously unseen samples.

You don't need to train the CNN if you use DeepSom to call somatic variants for samples of acute myeloid leukemia (LAML),  bladder urothelial cancer (BLCA), esophageal adenocarcinoma (ESAD), liver cancer (LINC), or gastric cancer (GACA).  Ready-to-use models for these cancers can be found in [models](models/)

An example CNN run in train mode:

```
python nn.py \
--train_dataset './train/train_dataset_name/tensors/train_imgb.lst' \
--output_dir './train/train_dataset_name/results/' \
--tensor_width 150 \
--tensor_height 70 \
--max_depth 150 \
--val_fraction 0.0 \
--save_each 10
```

`--tensor_width` --- width of the variant tensor

`--tensor_height` --- maximal height of the variant tensor. The VAF resolution of 1/`tensor_height` is guaranteed *only if the same height was used for tensor generation*, otherwise tensors with a larger height are just cropped.

`--max_depth` --- maximal read depth (used to normalize the flanking regions feature). Since the read depth distribution
often has a long right tail, we recommend using the 99th quantile of this distribution as `--max_depth` (about 2x median coverage)

`--val_fraction` --- percentage of input imgb batches used for validation and not for training. After each training epoch, the CNN performance is evaluated on the validation set. CNN predictions on the validation set are saved to
`output_dir/predictions/validation_epoch_N.vcf`.

`--save_each` --- how often model and optimizer weights as well as predictions on train dataset should be saved.

See `python nn.py --help` for more training options.

The model weights are saved in `output_dir/weights/epoch_N_weights_model`, the optimizer weights are saved in `output_dir/weights/epoch_N_weights_optimizer`. Optimizer weights are only required when one needs to resume training.

One can combine train and test/inference modes by using both `--train_dataset` and `test_dataset` parameters.

Training for 120 000 SNP and 20 000 INDEL variants (`tensor_width`=150; `tensor_height`=70) requires about 1.5Gb RAM and takes about 2.5h on NVIDIA Tesla 100V.

# Using the CNN output to compute probabilities and classify variants

For any given variant, the CNN outputs a continuous score ![equation](https://latex.codecogs.com/svg.image?s)
between 0 and 1. The higher ![equation](https://latex.codecogs.com/svg.image?s)
, the higher the probability ![equation](https://latex.codecogs.com/svg.image?p_%7Bsom%7D) that the variant is somatic. Variant classification is performed by imposing a threshold on ![equation](https://latex.codecogs.com/svg.image?s) s.t. all variants with ![equation](https://latex.codecogs.com/svg.image?s%3Es_%7Bthr%7D) are considered somatic. This threshold can be chosen based on
the [ROC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) or [Precision-Recall](https://en.wikipedia.org/wiki/Precision_and_recall) curve. The ROC curve is a plot of true positive rate (TPR) against false positive rate (FPR) w.r.t. somatic variants. The Precision-Recall curve is a plot of recall against precision w.r.t. somatic variants. For example, using the Precision-Recall curve, one can choose the threshold that maximizes the [f1-score](https://en.wikipedia.org/wiki/F-score) or guarantees a given level of recall.  Note that variants used for choosing ![equation](https://latex.codecogs.com/svg.image?s_%7Bthr%7D) should be different from those used for CNN training.

Alternatively, the classification threshold can be chosen based on the corresponding probability ![equation](https://latex.codecogs.com/svg.image?p_%7Bsom%7D). The relation between ![equation](https://latex.codecogs.com/svg.image?s) and ![equation](https://latex.codecogs.com/svg.image?p_%7Bsom%7D) is non-linear.

To compute ![equation](https://latex.codecogs.com/svg.image?p_%7Bsom%7D) based on ![equation](https://latex.codecogs.com/svg.image?s), one needs to calibrate the CNN output. For calibration, one runs the pre-trained CNN on test variants. Then, the CNN output values are binned s.t. there are at least 20 values per bin. The probability that a variant whose score ends up in bin ![equation](https://latex.codecogs.com/svg.image?s_%7Bi%7D) is somatic is given by the Bayes formula:

![equation](https://latex.codecogs.com/svg.image?p_%7Bsom%7D(s%5Csubset%20s_i)=%5Cfrac%7BP(s%5Csubset%20s_i%7Csom)%5Ctimes%20N_%7Bsom%7D%7D%7BP(s%5Csubset%20s_i%7Csom)%5Ctimes%20N_%7Bsom%7D%20&plus;%20P(s%5Csubset%20s_i%7Cneg)%5Ctimes%20N_%7Bneg%7D%7D)

where ![equation](https://latex.codecogs.com/svg.image?N_%7Bsom%7D) is the number of somatic variants per WGS sample,
![equation](https://latex.codecogs.com/svg.image?N_%7Bneg%7D) is the number of germline variants and artefacts per WGS sample, ![equation](https://latex.codecogs.com/svg.image?P(s%5Csubset%20s_i%7Csom)%20) is the fraction of true somatic variants at the input that end up in bin ![equation](https://latex.codecogs.com/svg.image?s_%7Bi%7D), ![equation](https://latex.codecogs.com/svg.image?P(s%5Csubset%20s_i%7Cneg)%20)  is the fraction of true germline variants and artefacts at the input that end up in bin ![equation](https://latex.codecogs.com/svg.image?s_%7Bi%7D).

Note that the ROC and Precision-Recall curves as well as probabilities must be computed separately for SNP and INDEL variants.
