# Test run in Inference mode

To provide a practical example on how to run the CNN in inference mode, we prepared a short example set of variants extracted from the [BLCA-US dataset](https://dcc.icgc.org/projects/BLCA-US). Data was preprocessed as described [here](../../dataprep/). In particular, variants were annotated for flanking regions, all variants with a gnomAD record were removed, and per-sample VCF files were merged. Based on the final VCF file
[BLCA-US_example_20220901.vcf](../test/BLCA-US/BLCA-US_example_20220901.vcf), a .imgb batch was generated.

The CNN can then be run using the following command:

```
conda activate DeepSom-cnn

python ../../cnn/nn.py --tensor_width=150 --tensor_height=70 --test_dataset=test_imgb.lst --model_weight=../../cnn/models/gnomAD_thr_0/BLCA-US_gnomAD_thr_0_epoch_20_weights_model
```

The final predictions will be saved in `./predictions/final_predictions.vcf`.
