# Neural Network Classifier for calling somatic SNPs without a matched normal.

The Neural Network Classifier (NNC) can be used in three modes:

1. Train mode. During training, the NNC learns relationships between variants and variant classes.

2. Evaluation (test) mode. The performance of the trained model is evaluated using an independent test set.
Such metrics as ROC AUC or test loss can be calculated.

3. Inference mode. The probability for a given variant to be somatic is computed based on the NNC output.

See  [dataprep](dataprep/) on how to prepare data for the NNC.

See  [NNC](NNC/) on how to run the NNC.
