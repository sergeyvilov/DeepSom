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
--negative_ds './datasets/dataset_name/tensors/train/train_neg_imgb.lst' \
--positive_ds './datasets/dataset_name/tensors/train/train_pos_imgb.lst' \
--output_dir './datasets/dataset_name/checkpoints' \
--tensor_width 150 \
--tensor_height 70 \
--val_fraction 0.0 \
--resample_train 'upsampling' \
--save_each 5
```

`--tensor_width` --- width of the variant tensor

`--tensor_height` --- maximal height of the variant tensor. Variant tensors with more reads will be cropped.

`--output_dir` --- folder to save model and optimizer weights as well as NNC prediction scores computed on the validation (test) set.

`--val_fraction` --- percentage of input variants used for validation and not for training. After each training epoch, the NNC performance is evaluated on the validation set.

`--resample_train` --- use 'upsample' to create a balanced train set by upsampling the class with the lower number of variants, 'downsample' to create a balanced train set by downsampling the class with the higher number of variants, 'None' if resampling of the train set is not needed.

`--save_each` --- how often model and optimizer weights should be saved on the disk.

See `python nn.py --help` for more training options.

Training for 70 000 positive-class and 70 000 negative-class tensors (`tensor_width`=150; `tensor_height`=70) takes about 20h on NVIDIA Tesla 100V.

## Evaluation (test) mode.

An example run of the NNC in evaluation (test) mode:

```
python nn.py \
--negative_ds './datasets/dataset_name/tensors/test/test_neg_imgb.lst' \
--positive_ds './datasets/dataset_name/tensors/test/test_pos_imgb.lst' \
--output_dir './datasets/dataset_name/test' \
--load_weights 1 \
--config_start_base './datasets/dataset_name/checkpoints/epoch_20_weights' \
--tensor_width 150 \
--tensor_height 70 \
--val_fraction 1.0
```
So, it looks like the train mode, but all the tensors are used for evaluation (`--val_fraction 1.0`).
Parameters `--load_weights` and `--config_start_base` are used to load the weights of a pretrained model to perform evaluation.

## Inference mode.

An example run of the NNC in inference mode:

```
python nn.py \
--inference_ds './projects/project_name/to_classify/tensors/inference_imgb.lst' \
--inference_mode 1 \
--output_dir './inference/projects/project_name/to_classify/dataset_name' \
--load_weights 1 \
--config_start_base './datasets/dataset_name/checkpoints/epoch_20_weights' \
--tensor_width 150 \
--tensor_height 70
```

NNC output scores for variants in `--inference_ds` will be written to an `inference.csv` file in `--output_dir`.
Each line of `inference.csv` corresponds to a single SNP variant. Use `variants.csv.gz` from the dataset tensors dir to identify
variants in `--inference_ds`.

# Using NNC output to compute probabilities and classify variants

For any given variant, the NNC outputs a continuous score ![equation](https://latex.codecogs.com/svg.image?s)
between 0 and 1. The higher ![equation](https://latex.codecogs.com/svg.image?s)
, the higher the probability ![equation](https://latex.codecogs.com/svg.image?p_%7Bsom%7D) that the variant is somatic. The relation between ![equation](https://latex.codecogs.com/svg.image?s) and ![equation](https://latex.codecogs.com/svg.image?p_%7Bsom%7D) is non-linear.

To compute ![equation](https://latex.codecogs.com/svg.image?p_%7Bsom%7D) based on ![equation](https://latex.codecogs.com/svg.image?s), one needs to calibrate the NNC output. For calibration, one runs the pre-trained NNC on test SNPs. Then, the NNC output values are binned s.t. there are at least 20 values per bin. The probability that a variant whose score ends up in bin ![equation](https://latex.codecogs.com/svg.image?s_%7Bi%7D) is somatic is given by the Bayes formula:

![equation](https://latex.codecogs.com/svg.image?p_%7Bsom%7D(s%5Csubset%20s_i)=%5Cfrac%7BP(s%5Csubset%20s_i%7Csom)%5Ctimes%20N_%7Bsom%7D%7D%7BP(s%5Csubset%20s_i%7Csom)%5Ctimes%20N_%7Bsom%7D%20&plus;%20P(s%5Csubset%20s_i%7Cneg)%5Ctimes%20N_%7Bneg%7D%7D)

where ![equation](https://latex.codecogs.com/svg.image?N_%7Bsom%7D) is the number of somatic variants per WGS sample,
![equation](https://latex.codecogs.com/svg.image?N_%7Bneg%7D) is the number of germline variants and artefacts per WGS sample, ![equation](https://latex.codecogs.com/svg.image?P(s%5Csubset%20s_i%7Csom)%20) is the fraction of true somatic variants at the input that end up in bin ![equation](https://latex.codecogs.com/svg.image?s_%7Bi%7D), ![equation](https://latex.codecogs.com/svg.image?P(s%5Csubset%20s_i%7Cneg)%20)  is the fraction of true germline variants and artefacts at the input that end up in bin ![equation](https://latex.codecogs.com/svg.image?s_%7Bi%7D).


Variant classification is performed by imposing a threshold on ![equation](https://latex.codecogs.com/svg.image?s) s.t. all variants with ![equation](https://latex.codecogs.com/svg.image?s%3Es_%7Bthr%7D) are considered somatic. This threshold can be chosen based on the corresponding probability ![equation](https://latex.codecogs.com/svg.image?p_%7Bsom%7D). Alternatively, ![equation](https://latex.codecogs.com/svg.image?s_%7Bthr%7D) can be chosen based on the ROC curve. The ROC curve is a plot of the true positive rate (TPR) against the false positive rate (FPR) w.r.t. somatic variants. When ![equation](https://latex.codecogs.com/svg.image?s_%7Bthr%7D) increases the operating point on the ROC curve moves from the upper right to the lower left quadrant. In any case, variants used for choosing ![equation](https://latex.codecogs.com/svg.image?s_%7Bthr%7D) should be different from those used for NNC training.
