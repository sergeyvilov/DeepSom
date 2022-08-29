# DeepSom. A CNN - based pipeline for calling somatic SNP and INDEL variants without a matched normal

DeepSom can operate in three modes:

- Inference mode. In this mode, the convolutional neural network (CNN) assigns a pseudoprobability score to each candidate variant which can then be classified as somatic or non-somatic. This is the main operating mode.

- Test(evaluation) mode. In this mode, we use variants with known labels (somatic/non-somatic) to evaluate DeepSom performance.

- Train mode. In this  mode, the CNN learns relationships between variant tensors and variant labels (somatic/non-somatic). This results in a model which can be used in inference or evaluation.

See [dataprep](dataprep/) on how to prepare data for the CNN.

See [cnn](cnn/) on how to run the CNN.
