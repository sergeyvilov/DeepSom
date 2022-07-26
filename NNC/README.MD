# Neural Network Classifier.

## Setup

Install  [miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

Then run:

```
conda env create -f environment.yml
conda activate nnc
```

## Train mode.

An example run of the NNC in train mode:

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

`--tensor_height` --- maximal height of the variant tensor. Variant tensors with more reads will be cropped.

`--max_depth` --- maximal read depth (used to normalize the flanking regions feature). Since the read depth distribution
often has a long right tail, we recommend using the 99th quantile of this distribution as `--max_depth` (about 2x median coverage)

`--output_dir` --- folder to save model and optimizer weights as well as model predictions.

`--val_fraction` --- percentage of input imgb batches used for validation and not for training. After each training epoch, the NNC performance is evaluated on the validation set. NNC predictions on the validation set are saved to
`output_dir/predictions/validation_epoch_N.vcf`.

`--save_each` --- how often model and optimizer weights as well as predictions on train dataset should be saved on the disk.
The model weights are saved in `output_dir/weights/epoch_N_weights_model`, the optimizer weights are saved in `output_dir/weights/epoch_N_weights_optimizer`. Optimizer weights are only required when one needs to resume training.

See `python nn.py --help` for more training options.

Training for 120 000 SNP and 20 000 INDEL variants (`tensor_width`=150; `tensor_height`=70) takes about 2.5h on NVIDIA Tesla 100V.

## Test/inference mode.

This mode can be used for evaluation of the model (when true variant labels are known) as well as for inference (when true variant labels aren't known and the model is used for variant classification).

An example run of the NNC in test/inference mode:

```
python nn.py \
--test_dataset './inference/project_name/wgs_sample_name/tensors/wgs_sample_name_imgb.lst' \
--output_dir './inference/project_name/wgs_sample_name/results/' \
--start_weight_model './train/train_dataset_name/results/weights/epoch_N_weights_model' \
--tensor_width 150 \
--tensor_height 70 \
--max_depth 150 \
```

`--start_weight_model` --- path to the weights of a pretrained model.

`--max_depth`, `--tensor_width`, `--tensor_height` --- should correspond to values used for training.

Predictions are saved to `output_dir/predictions/final_predictions.vcf`, where the NNC score is stored under `nnc_score` record in the INFO field.

One can combine train and test/inference modes by using both `--train_dataset` and `test_dataset` options.

# Using NNC output to compute probabilities and classify variants

For any given variant, the NNC outputs a continuous score ![equation](https://latex.codecogs.com/svg.image?s)
between 0 and 1. The higher ![equation](https://latex.codecogs.com/svg.image?s)
, the higher the probability ![equation](https://latex.codecogs.com/svg.image?p_%7Bsom%7D) that the variant is somatic. Variant classification is performed by imposing a threshold on ![equation](https://latex.codecogs.com/svg.image?s) s.t. all variants with ![equation](https://latex.codecogs.com/svg.image?s%3Es_%7Bthr%7D) are considered somatic. This threshold can be chosen based on
the [ROC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) or [Precision-Recall](https://en.wikipedia.org/wiki/Precision_and_recall) curve. The ROC curve is a plot of true positive rate (TPR) against false positive rate (FPR) w.r.t. somatic variants. The Precision-Recall curve is a plot of recall against precision w.r.t. somatic variants. For example, using the Precision-Recall curve, one can choose the threshold that maximizes the [f1-score](https://en.wikipedia.org/wiki/F-score).  In any case, variants used for choosing ![equation](https://latex.codecogs.com/svg.image?s_%7Bthr%7D) should be different from those used for NNC training.

Alternatively, the classification threshold can be chosen based on the corresponding probability ![equation](https://latex.codecogs.com/svg.image?p_%7Bsom%7D). The relation between ![equation](https://latex.codecogs.com/svg.image?s) and ![equation](https://latex.codecogs.com/svg.image?p_%7Bsom%7D) is non-linear.

To compute ![equation](https://latex.codecogs.com/svg.image?p_%7Bsom%7D) based on ![equation](https://latex.codecogs.com/svg.image?s), one needs to calibrate the NNC output. For calibration, one runs the pre-trained NNC on test variants. Then, the NNC output values are binned s.t. there are at least 20 values per bin. The probability that a variant whose score ends up in bin ![equation](https://latex.codecogs.com/svg.image?s_%7Bi%7D) is somatic is given by the Bayes formula:

![equation](https://latex.codecogs.com/svg.image?p_%7Bsom%7D(s%5Csubset%20s_i)=%5Cfrac%7BP(s%5Csubset%20s_i%7Csom)%5Ctimes%20N_%7Bsom%7D%7D%7BP(s%5Csubset%20s_i%7Csom)%5Ctimes%20N_%7Bsom%7D%20&plus;%20P(s%5Csubset%20s_i%7Cneg)%5Ctimes%20N_%7Bneg%7D%7D)

where ![equation](https://latex.codecogs.com/svg.image?N_%7Bsom%7D) is the number of somatic variants per WGS sample,
![equation](https://latex.codecogs.com/svg.image?N_%7Bneg%7D) is the number of germline variants and artefacts per WGS sample, ![equation](https://latex.codecogs.com/svg.image?P(s%5Csubset%20s_i%7Csom)%20) is the fraction of true somatic variants at the input that end up in bin ![equation](https://latex.codecogs.com/svg.image?s_%7Bi%7D), ![equation](https://latex.codecogs.com/svg.image?P(s%5Csubset%20s_i%7Cneg)%20)  is the fraction of true germline variants and artefacts at the input that end up in bin ![equation](https://latex.codecogs.com/svg.image?s_%7Bi%7D).

Note that the ROC and Precision-Recall curves as well as probabilities must be computed separately for SNP and INDEL variants.
