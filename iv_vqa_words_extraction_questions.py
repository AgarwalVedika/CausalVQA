import json
import time
import config
from vocab_mapping import nouns_matched_COCO, nouns_matched_COCO_stuff ,get_words


## just getting the list of words- index doesnt matter here
with open('coco-labels/coco_labels_80_classes.txt') as f:
    # return the split results, which is all the words in the file.
    coco_80 = f.read().splitlines()  ###coco 80 classes

# with open('all_coco.txt') as f:
#     # return the split results, which is all the words in the file.
#     coco_stuff_91= f.read().splitlines()      ##coco_stuff 91 classes


ann_coco_file_val = config.coco_ann_dir + 'instances_val2014.json'
ann_coco_file_train = config.coco_ann_dir + 'instances_train2014.json'
with open(ann_coco_file_train) as f:
    ann_coco_1 = json.load(f)['categories']

id_list = []
class_list = []
for cate in ann_coco_1:
    id_list.append(cate['id'])
    class_list.append(cate['name'])



def ready_json(question_file, filename, nouns_list):
    start = time.time();
    a1 = nouns_matched_COCO(nouns_list, coco_80, id_list, class_list)  ##here attach class_id for person, if man/woman was there
    # a3 = nouns_matched_COCO_stuff(nouns_list, coco_stuff_91)
    abcd = []
    file_data = {}

    with open(question_file) as f:
        questions = json.load(f)['questions']  ##read json


    for n_q, q in enumerate(questions):
        # print(n_q)
        abcd.append({"image_id": q['image_id'],
                     "question": q['question'],
                     "question_id": q['question_id'],
                     "nouns_q_COCO": a1[n_q]})
        # "nouns_q_COCO_stuff": a3[n_q]})
        # "nouns_q": nouns_list[n_q],

    file_data['questions'] = abcd
    with open(filename, 'w') as outfile_val1:
        json.dump(file_data, outfile_val1)
    print(time.time() - start)


question_file_1 = config.vqa_q_dir + "v2_OpenEnded_mscoco_train2014_questions.json"
question_file_2 = config.vqa_q_dir + "Questions/v2_OpenEnded_mscoco_val2014_questions.json"
question_file_3 = config.vqa_q_dir + "v2_OpenEnded_mscoco_test-dev2015_questions.json"
question_file_4 = config.vqa_q_dir + "v2_OpenEnded_mscoco_test2015_questions.json"


nouns_list_1 = get_words(question_file_1)
ready_json(question_file_1,'tagged_train2014_questions.json', nouns_list_1)


nouns_list_2 = get_words(question_file_2)
ready_json(question_file_2, 'tagged_val2014_questions.json', nouns_list_2)


# nouns_list_3 = get_words(question_file_3)
# ready_json(question_file_3, 'tagged_test_dev_questions.json', nouns_list_3)

# nouns_list_4 = get_words(question_file_4)
# ready_json(question_file_4, 'tagged_test_questions.json', nouns_list_4)
