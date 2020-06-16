import json
import time
import itertools
from torch.utils.data import Dataset
import pickle
import ipdb
import config
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from pycocotools.coco import COCO
from skimage.morphology import disk, square
from collections import defaultdict
import scipy
selem = disk(2)
square_dil = square(5)




cosider_flip_too = 0
all_10_ans_common=1
ann_coco_file_val = config.coco_ann_dir  + "instances_val2014.json"
ann_coco_file_train = config.coco_ann_dir  + "instances_train2014.json"
COCO_val = COCO(ann_coco_file_val)
COCO_train = COCO(ann_coco_file_train)


with open(ann_coco_file_val) as f:
    ann_coco_1 = json.load(f)['categories']
coco_dict = {}
for i, det in enumerate(ann_coco_1):
    coco_dict[det['id']] = det['name']


class VQADataset_custom(Dataset):
    """VQA dataset"""

    def __init__(self, coco_pkl_file, ques_ann_path, ans_ann_path, mode):
        """
        Args:
            ques_ann (string): Path to the json file with ques_annotations.
            ans_ann (string): Path to the json file with ans_annotations.
        """

        self.coco_details = pickle.load(open(coco_pkl_file, 'rb'))['area_and_intersection']
        self.questions = json.load(open(ques_ann_path, 'r'))[
            'questions']  ###or self.questions = load_vocab(ques_ann_path)
        self.answers = json.load(open(ans_ann_path, 'r'))['answers']
        self.mode = mode

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        """Returns ONE data pair-image,match_coco_objects"""

        q = self.questions
        area_inter = self.coco_details
        ans = self.answers

        assert q[idx]['image_id'] == area_inter[idx]['image_id'] == ans[idx]['image_id']

        classes_img = area_inter[idx]['classes_img']
        percent_area_per_catId_all_inst = area_inter[idx]['percent_area_per_catId_all_inst']
        percent_area_per_catId_max_inst = area_inter[idx]['percent_area_per_catId_max_inst']
        if_intersect_overlap_sq5 = area_inter[idx]['if_intersect_overlap_sq5']  # or "if_intersect_overlap_default"
        # if_intersect_overlap_default = area_inter[idx]['if_intersect_overlap_default']
        flipped_if_intersect_overlap_sq5 = area_inter[idx]['flipped_if_intersect_overlap_sq5']
        iou_mask_flipped_mask = area_inter[idx]['iou_mask_flipped_mask']

        # print('Reading image data')
        img_id = self.questions[idx]['image_id']
        # print(img_id)

        question = self.questions[idx]['question']
        question_id = self.questions[idx]['question_id']
        # nouns_q = questions[idx]['nouns_q']
        # nouns_q_coco_stuff = questions[idx]['nouns_q_COCO_stuff']

        # print('Reading nouns data')
        nouns_q_coco = self.questions[idx]['nouns_q_COCO']
        nouns_ans = self.answers[idx]['ans_match_COCO']
        # print(img_id, nouns_img, nouns_q_coco, nouns_ans )

        answers = [i['answer'] for i in self.answers[idx]['answers']]

        question_type = self.answers[idx]['question_type']
        multipe_choice_answer = self.answers[idx]['multipe_choice_answer']
        answers_std = self.answers[idx]['answers']
        question_id2 = self.answers[idx]['question_id']
        assert question_id == question_id2
        answer_type = self.answers[idx]['answer_type']


        if cosider_flip_too:
            return answers, classes_img, nouns_q_coco, nouns_ans, img_id, question_id, question, \
                   percent_area_per_catId_all_inst, if_intersect_overlap_sq5, percent_area_per_catId_max_inst, \
                   flipped_if_intersect_overlap_sq5, iou_mask_flipped_mask, question_type, answer_type, multipe_choice_answer, answers_std
        else:
            return answers, classes_img, nouns_q_coco, nouns_ans, img_id, question_id, question, \
                   percent_area_per_catId_all_inst, if_intersect_overlap_sq5, percent_area_per_catId_max_inst, \
                   question_type, answer_type, multipe_choice_answer, answers_std

def count_dict(given_list):
    nouns_dict = {}
    for i in given_list:
        if i in nouns_dict:
            nouns_dict[i]+=1
        else:
            nouns_dict[i] =1
    return nouns_dict


coco_val_pkl = './../iv_vqa_generation/coco_areas_and_intersection/coco_vqa_val2014.json'
coco_train_pkl = './../iv_vqa_generation/coco_areas_and_intersection/coco_vqa_train2014.json'


def prep_qa_pkl(mode, dataset, filename, filename_ans, id_area_overlap_dict_inst_id, ques_id_id, overlap_thresh):
    start = time.time()
    abcd = []
    file_data = {}
    abcd_standard_json = []
    file_data_standard_json = {}
    abcd_standard_ans_json = []
    file_data_standard_ans_json = {}

    for i in range(len(dataset)):

        answers, classes_img, nouns_q, nouns_ans, img_id, question_id, question, \
        percent_area_per_catId_all_inst, if_intersect_overlap, percent_area_per_catId_max_inst, \
        question_type, answer_type, multiple_choice_answer, answers_std = dataset[i]

        classes_img_set = sorted(list(set(classes_img)))

        nouns_q_str = [coco_dict[i] for i in nouns_q]
        classes_img_str = [coco_dict[i] for i in classes_img]
        count_nouns_q = count_dict(nouns_q)
        count_classes_img = count_dict(classes_img)
        allowed_char_in_ans = [str(i) for i in range(10)]  # ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        ans_cond = [True if i in allowed_char_in_ans else False for i in answers[0]]
        final_ans_cond = True
        if False in ans_cond:
            final_ans_cond = False
        # ques_condition_neg ='number of' not in question and 'many' not in question and 'what number is' not in question\
        # and "What's the number" not in question and  'What number is' not in question
        if 'many' in question:
            if 'person' in count_nouns_q.keys():
                count_nouns_q['person'] = count_nouns_q['person'] - 1
                if count_nouns_q['person'] == 0:
                    del count_nouns_q['person']
            elif 1 in count_nouns_q.keys():
                count_nouns_q[1] = count_nouns_q[1] - 1
                if count_nouns_q[1] == 0:
                    del count_nouns_q[1]

        ques_condition = 'many' in question or 'number of' in question

        # details = i['id'], i['area_occupied'], i['overlap_score_with_rest'], i['image_id']

        if len(set(answers)) == 1 and answer_type == 'number' and final_ans_cond and ques_condition:
            key = str(question_id) + '_' + str(img_id)
            if key in ques_id_id:
                for diff_ann_inst_id in ques_id_id[key]:
                    specific_details = id_area_overlap_dict_inst_id[diff_ann_inst_id]
                    os = specific_details[2]  ## overlap score
                    if (isinstance(os, str) and os == 'only_candidate_instance') or (isinstance(os, float) and os <= overlap_thresh) :
                        #print(specific_details)
                        new_i_id = str(img_id).zfill(12) + '_' + str(diff_ann_inst_id).zfill(12)
                        abcd.append({"image_id": new_i_id,
                                     "question": question,
                                     "question_id": question_id,
                                     "area_occupied": specific_details[1],
                                     "overlap_score_with_rest": specific_details[2]})

                        abcd_standard_json.append({"image_id": new_i_id,
                                     "question": question,
                                     "question_id": question_id})

                        ch_answer = [i['answer'] for i in answers_std]
                        ch_answer_conf = [i['answer_confidence'] for i in answers_std]
                        ch_answer_id = [i['answer_id'] for i in answers_std]

                        new_ch_answer = [str(int(i)-1) for i in ch_answer]   ### correponding answer reduces by 1 (one object removed at a time) ## can be figured by counting
                        #'_' in new_i_id

                        new_answers_std = []
                        for i in range(len(new_ch_answer)):
                            mini_ans = {}
                            mini_ans['answer'] = new_ch_answer[i]
                            mini_ans['answer_confidence'] = ch_answer_conf[i]
                            mini_ans['answer_id'] = ch_answer_id[i]
                            new_answers_std.append(mini_ans)

                        abcd_standard_ans_json.append({"question_type": question_type,
                             "multiple_choice_answer": multiple_choice_answer,
                             "answers": new_answers_std,
                             "image_id": new_i_id,
                             "answer_type": answer_type,
                             "question_id": question_id})


    file_data['questions'] = abcd
    file_data_standard_json['questions'] = abcd_standard_json
    file_data_standard_ans_json['annotations'] = abcd_standard_ans_json

    with open(filename, 'wb') as outfile_val1:
        pickle.dump(file_data, outfile_val1, pickle.HIGHEST_PROTOCOL)

    filename_json = filename.strip('pickle') + 'json'
    with open(filename_json, 'w') as outfile_val2:
        json.dump(file_data_standard_json, outfile_val2)

    with open(filename_ans, 'w') as outfile_val2:
        json.dump(file_data_standard_ans_json, outfile_val2)

    print(time.time() - start)





def final_prep(mode, coco_pkl, given_area_thresh, given_overlap_thresh):  # mode='val2014'  string
    question_path = './../iv_vqa_generation/tagged_' + mode + '_questions.json'  ## corresponds to question.json file
    answer_path = './../iv_vqa_generation/tagged_' + mode + '_answers.json'

    root_dir_ques = config.iv_q_dir
    os.makedirs(root_dir_ques, exist_ok=True)

    root_dir_ans = config.iv_a_dir
    os.makedirs(root_dir_ans, exist_ok=True)

    res_file_ques = 'v2_OpenEnded_mscoco_' + mode + '_questions.pickle'
    res_file_ans = 'v2_mscoco_' + mode + '_annotations.json'

    # id_area_overlap
    with open(mode + 'coco_counting_id_area_overlap_only_one_considered_at_a_time.pickle', 'rb') as file:
        id_area_overlap = pickle.load(file)

    dict_id_area_overlap_dict_inst_id = {}
    for i in id_area_overlap:
        key = i['id']
        details = i['id'], i['area_occupied'], i['overlap_score_with_rest'], i['image_id']
        dict_id_area_overlap_dict_inst_id[key] = details

    dict_ques_id_id = defaultdict(list)
    for i in id_area_overlap:
        key = str(i['question_id']) + '_' + str(i['image_id'])
        details = i['id']
        if not details in dict_ques_id_id[key]:
            dict_ques_id_id[key].append(details)

    start = time.time()
    dataset_mode = VQADataset_custom(coco_pkl, question_path, answer_path, mode)
    print(time.time() - start)

    ## prep_q_pkl(dataset, filename, id_area_overlap_dict_inst_id, ques_id_id,  overlap_thresh)
    prep_qa_pkl(mode, dataset_mode, os.path.join(root_dir_ques, res_file_ques), os.path.join(root_dir_ans, res_file_ans),dict_id_area_overlap_dict_inst_id, dict_ques_id_id, given_overlap_thresh)

    # sample check to make sure
    #print(dataset_mode[0])
    with open(os.path.join(root_dir_ques, res_file_ques), 'rb') as f:
        edited_questions = pickle.load(f)['questions']
    # with open( os.path.join(root_dir, res_file)) as f:
    #    edited_questions = json.load(f)['questions']
    print(len(edited_questions))
    print(edited_questions[0])

    with open(os.path.join(root_dir_ans, res_file_ans), 'rb') as f:
        edited_answers = json.load(f)['annotations']
    print(edited_answers[0])

    assert len(edited_answers) == len(edited_questions)


final_prep('val2014', coco_val_pkl, 0.1, 0.0)
final_prep('train2014', coco_train_pkl, 0.1, 0.0)