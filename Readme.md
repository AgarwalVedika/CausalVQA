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

## Installation
Use Anaconda to set up an environment and install the dependencies listed in requirements.txt file
```
conda create --name <env> --file requirements.txt
```
Clone this repository with Git, and then enter the root directory of the repository:
```
git clone https://github.com/AgarwalVedika/CausalVQA.git && cd CausalVQA
```
Download this zip folder and extract everything to the CausalVQA directory:  
https://drive.google.com/file/d/16TFNJLMYdkR1LrFL4MwqmNkjOjPOokUs/view?usp=sharing

## Dataset Generation

We use a pre-trained object removal model. The model is available within the pre-trained models (within the zip folder downloaded above).  

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
Similarly we will generate the CV-VQA dataset, since words extraction was already done above- no need to repeat this step. 
```
cd ./../cv_vqa_generation
python cv_vqa_gen_images.py --input_mode train2014  #images
python cv_vqa_gen_images.py --input_mode val2014  #images
cv_vqa_gen_area_overlap_score.ipynb  #run the jupyter notebook to get area/overlap
python cv_vqa_gen_qa.py  #QA
```


## Analysis

We use 3 models: SNMN, SAAA and CNN+LSTM.  We train the models as decribed by the authors. 
Make sure to save the answer id, image id, question id and the softmax vector (this helps for visualization) for analysis.  
Keys used in analysis code to refer these are: 'ans_id', 'img_id', 'ques_id' and 'ss_vc'.

For analysis purpose- we select only those IQAs in original VQA v2 with uniform answer.
The entire validation set is split into 90:10 where the former is used for testing, latter for validation. 
One can find the original and edited QA files here in the testing folder (within the zip folder downloaded above.) 

For calculating flips ad accuracy, please edit the results paths in the config (Analysis section). Then run the following jupyter notebook.
```
flip_accuracy_cal_iv_vqa.ipynb # flip/acc for iv_vqa
flip_accuracy_cal_cv_vqa.ipynb # flip/acc for cv_vqa     
```

### Visualization- clicking tool
We also visualize different IQA (original and edited) on basis of the difference in their softmax vector. We build a simple matplotlib based clicking 
tool to achieve this. One might need to edit the paths in this code, however can re-use the clicking tool provided. 
```
python on_pick_all3.py  #iv_vqa
python on_pick_all3_counting_del1.py  #cv_vqa
```


## Acknowledgements
Object removal code is inspired from the Object removal GAN repository (https://github.com/rakshithShetty/adversarial-object-removal). 
To train SNMN, SAAA and CNN+LSTM, we follow the code repositories below. CNN+LSTM is built by modifying SAAA code. 
For SNMN: https://github.com/ronghanghu/snmn
For SAAA: https://github.com/Cyanogenoid/pytorch-vqa


## Miscellaneous
For CNN+LSTM/SAAA and training these models using data augmentation while enforcing different consistency losses: one can take a look at 
https://github.com/AgarwalVedika/pytorch-vqa. 
Kindly note- the code repository is not cleaned and path editing would be required. The code snippets sure will be helpful.  


 
