# vqa images
vqa_images_dir = "VQA_v2/Images/"
vqa_q_dir = "VQA_v2/Questions/"
vqa_a_dir = "VQA_v2/Annotations/"

## coco annotations
coco_ann_dir =  "coco/annotations/"
ann_coco_file_val = 'coco/annotations/instances_val2014.json'
ann_coco_file_train = 'coco/annotations/instances_train2014.json'

#pre-trained object rmeoval models
removal_model_256 = "pretrained_removal_models/checkpoint_stargan_coco_fulleditor_wgan_50pcUnion_msz32_withPmask_L1150_style3k_tv_nb4_512sz_sqznet_mgpu_b14_255_3483.pth.tar"  #eyeballing- better than 512
removal_model_512 = "pretrained_removal_models/checkpoint_stargan_coco_fulleditor_wgan_40pcUnion_msz32_withPmask_L1150_tv_nb4_256sz_sqznet_250_1951.pth.tar"

# iv_vqa
iv_images_dir = "iv_vqa/Images/"
iv_q_dir = "iv_vqa/Questions/"
iv_a_dir = "iv_vqa/Annotations"

# cv_vqa
cv_images_dir = "cv_vqa/Images/"
cv_q_dir = "cv_vqa/Questions/"
cv_a_dir = "cv_vqa/Annotations"


# Anlaysis: edit file paths here

# testing QA files for original VQA,
iv_qa_test_dir_orig = 'testing/iv_vqa/original'
iv_qa_test_dir_edit= 'testing/iv_vqa/edited'
cv_qa_test_dir_orig = 'testing/cv_vqa/original'
cv_qa_test_dir_edit= 'testing/cv_vqa/edited'

# SNMN
results_edit_val_snmn_iv_vqa = '/BS/vedika2/nobackup/snmn/exp_vqa/eval_outputs_vqa_v2/vqa_v2_scratch_train/vqa_v2_edited_val2014_vqa_v2_scratch_train_15000_results.pickle'
results_edit_val_snmn_cv_vqa = '/BS/vedika2/nobackup/snmn/exp_vqa/eval_outputs_vqa_v2/vqa_v2_scratch_train_25lr/vqa_v2_counting_del1_edited_val2014_vqa_v2_scratch_train_25lr_30_results.pickle'
results_val_snmn = '/BS/vedika2/nobackup/snmn/exp_vqa/eval_outputs_vqa_v2/vqa_v2_scratch_train/vqa_v2_val2014_vqa_v2_scratch_train_15000_results.pickle'
standard_vocab_ans_file_snmn = '/BS/vedika2/nobackup/snmn/exp_vqa/data/answers_vqa.txt'

# SAAA
results_edit_val_saaa_iv_vqa = '/BS/vedika3/nobackup/pytorch-vqa/logs/edit_val_with_attn_.pickle'
results_edit_val_saaa_cv_vqa = '/BS/vedika3/nobackup/pytorch-vqa/logs/edit_val_with_attn_counting.pickle'
results_val_saaa = '/BS/vedika3/nobackup/pytorch-vqa/logs/val_with_attn_.pickle'
standard_vocab_ans_file_saaa = '/BS/vedika3/nobackup/pytorch-vqa/vocab.json'

# CNN_LSTM
results_edit_val_cnn_lstm_iv_vqa = '/BS/vedika3/nobackup/pytorch-vqa/logs/edit_val_no_attn_.pickle'
results_edit_val_cnn_lstm_cv_vqa = '/BS/vedika3/nobackup/pytorch-vqa/logs/edit_val_no_attn_counting.pickle'
results_val_cnn_lstm = '/BS/vedika3/nobackup/pytorch-vqa/logs/val_no_attn_.pickle'
standard_vocab_ans_file_cnn_lstm = '/BS/vedika3/nobackup/pytorch-vqa/vocab.json'

