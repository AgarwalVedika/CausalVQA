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
python iv_vqa_gen_images.py --input_mode train2014  
python iv_vqa_gen_images.py --input_mode val2014  
```

For all the images in train/val, generate an exhaustive set of images with one object (all instances) removed at a time.

QA

please change the directory:
```
cd iv_vqa_gen_qa
```

First we do some vocab mapping and word extraction. Please note: vocab_mapping.py has a function mapping the VQA vocab to 80 COCO objects. 

```
python iv_vqa_nouns_extractor_questions.py
python iv_vqa_nouns_extractor_answers.py
```

Run the command below to calculate area/overlapping using gt coco segmentations. This is needed for creating final QA jsons files. 
```
python get_areas_overlap.py
```

Finally use the below two jupyter notebooks to prepare the QA json files.
```
ques: gen_ans_json.ipynb 
ans: gen_ques_json.ipynb
```

After this step, you will have all the images,questions and answers for IV-VQA dataset.


## Acknowledgements
Object removal code is inspired from the Object removal GAN repository (https://github.com/rakshithShetty/adversarial-object-removal).


