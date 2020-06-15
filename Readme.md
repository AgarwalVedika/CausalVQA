
## Work in progress

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

## Code coming soon- work in progress


Dataset synthesis:
Images IV VQA:
Images: What we do is that for all the images in coco: we remove the object and store them code: IV VQA: /BS/vedika2/nobackup/thesis/code/final_neat_script_batch_sz_1.py other related scripts for gan will be here as well: /BS/vedika2/nobackup/thesis/code/ and removal models used in /BS/vedika2/nobackup/thesis/code/removalmodels (Rakshith had sent a very nice email documenting step used, else some info in '/BS/vedika2/nobackup/thesis/code/pipeline.py')

QA vocab mapping- /BS/vedika2/nobackup/thesis/code/extract_utils.py - has a vocab mapping defined so first we do some vocab mapping and word extraction: ques: /BS/vedika2/nobackup/thesis/code/prep_json_standard_json_pkl/final_pipeline_nouns_extractor_questions.py ans: /BS/vedika2/nobackup/thesis/code/prep_json_standard_json_pkl/final_pipeline_nouns_extractor_answers.py

then a script to calculate area/overlapping using gt coco segmentations used for creating final QA jsons: /BS/vedika2/nobackup/thesis/code/get_areas_overlap.py

then there is final script to prep json-
ques: /BS/vedika2/nobackup/thesis/code/prep_json_standard_json_pkl/final_prep_ques_json_standard_for_model_eval.ipynb ans: /BS/vedika2/nobackup/thesis/code/prep_json_standard_json_pkl/final_prep_ans_json_standard.ipynb

CV VQA: images: /BS/vedika2/nobackup/thesis/code/COUNTING/final_neat_script_batch_sz_1_counting.py get area/overlap: /BS/vedika2/nobackup/thesis/code/COUNTING/get_areas_overlap_for_COUNTING.py QA: counting_prep_json_ques_ans.py

Analysis:
store answer id, image id, question id and softmax vector

calculating flips/accuracy code: SNMN_analysis_codes/ON_PICKING_barplot_final_average_CVPR_also_used_for_REBUTTAL.ipynb

so we need to tell them that they store softmax vector too if they want to visualize the plots/ do that clicking tool - IV VQA: /BS/vedika2/nobackup/thesis/code/SNMN_analysis_codes/on_pick_all3.py CV VQA: /BS/vedika2/nobackup/thesis/code/SNMN_analysis_codes/on_pick_all3_counting_del1.py

(Might be useful: there are more on_pick python scripts here- for different ques_types and DA comparison)
