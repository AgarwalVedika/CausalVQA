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
##Installation
Use Anaconda to set up an environment and install the dependencies listed in requirements.txt file
```
conda create --name <env> --file requirements.txt
```

Clone this repository with Git, and then enter the root directory of the repository:
```
git clone https://github.com/AgarwalVedika/CausalVQA.git && cd CausalVQA
```

## Dataset Generation

We use a pre-trained object removal model. Please find the 512x512 removal GAN used by us here: https://drive.google.com/file/d/1dIq-AGcUYSTf6QsOdvSoEJFcJk7MUxed/view?usp=sharing. Make sure it is stored in pre_removal_models folder.

### Generating IV-VQA dataset:

Images: 
The below two commands will generate an exhaustive set of images with one object (all instances) removed at a time for all the images in train/val.
```
cd iv_vqa_generation
python iv_vqa_gen_images.py --input_mode train2014  
python iv_vqa_gen_images.py --input_mode val2014  
```

QA:
1. First we do some vocab mapping and word extraction. Please note: vocab_mapping.py has a function mapping the VQA vocab to 80 COCO objects. 
```
python vqa_nouns_extractor_questions.py
python vqa_nouns_extractor_answers.py
```

2. Run the command below to calculate area/overlapping using gt coco segmentations. This is needed for creating final QA jsons files. 
```
python get_areas_overlap.py
```

3. Finally use the below two jupyter notebooks to prepare the QA json files.
```
questions: gen_ans_json.ipynb 
answers: gen_ques_json.ipynb
```

After this step, you will have all the images,questions and answers for IV-VQA dataset.


### Generating CV-VQA dataset:
Similarly we will generate the CV-VQA dataset, since words extraction was already doen above- no need to repeat this step. 
```
cd ./../cv_vqa_generation
python cv_vqa_gen_images.py --input_mode train2014  #images
python cv_vqa_gen_images.py --input_mode val2014  #images
cv_vqa_gen_area_overlap_score.ipynb  #run the jupyter notebook to get area/overlap
python cv_vqa_gen_qa.py  #QA
```


## Analysis: Code coming soon


## Acknowledgements
Object removal code is inspired from the Object removal GAN repository (https://github.com/rakshithShetty/adversarial-object-removal).


