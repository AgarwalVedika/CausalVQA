Repository for the paper "Towards Causal VQA: Revealing and Reducing Spurious Correlations by Invariant and Covariant Semantic Editing"

# Bibtex

~~~~~~~~~~~~~~~~
@inproceedings{agarwal2020towards,
  title={Towards Causal VQA: Revealing and Reducing Spurious Correlations by Invariant and Covariant Semantic Editing}
  author={Agarwal, Vedika and Shetty, Rakshith and Fritz, Mario},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
~~~~~~~~~~~~~~~~

## Dataset Generation

We use a pre-trained object removal model (https://github.com/rakshithShetty/adversarial-object-removal). Please find the 512x512 removal GAN used by us here: <GIVE LINK- GOOGLE DRIVE?>. Make sure it is stored in pre_removal_models folder.

1. Generating IV-VQA dataset:

```
python iv_image_gen.py --input_mode train2014  
python iv_image_gen.py --input_mode val2014  
```

For all the images in train/val, generate an exhaustive set of images with one object (all instances) removed at a time.


## Acknowledgements
Object removal code is inspired from the Object removal GAN repository (https://github.com/rakshithShetty/adversarial-object-removal).


