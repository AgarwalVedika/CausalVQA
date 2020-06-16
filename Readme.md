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

We use a pre-trained object removal model. Please find the 512x512 removal GAN used by us here: https://drive.google.com/file/d/1dIq-AGcUYSTf6QsOdvSoEJFcJk7MUxed/view?usp=sharing. Make sure it is stored in pre_removal_models folder.

1. Generating IV-VQA dataset:

Images
```
python iv_image_gen.py --input_mode train2014  
python iv_image_gen.py --input_mode val2014  
```

For all the images in train/val, generate an exhaustive set of images with one object (all instances) removed at a time.

QA
First we do some vocab mapping and word extraction. Please note: vocab_mapping.py has a function mapping the VQA vocab to 80 COCO objects. 

```
python iv_vqa_nouns_extractor_questions.py
python iv_vqa_nouns_extractor_answers.py
```

then a script to calculate area/overlapping using gt coco segmentations used for creating final QA jsons: 
/BS/vedika2/nobackup/thesis/code/get_areas_overlap.py



## Acknowledgements
Object removal code is inspired from the Object removal GAN repository (https://github.com/rakshithShetty/adversarial-object-removal).


