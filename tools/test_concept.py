import pickle
import json
import numpy as np

input_dir = "/home/wyf/open_source_dataset/artemis_dataset/3.29/artemis_caption_anno_train.pkl"
datalist = pickle.load(open(input_dir, 'rb'), encoding='bytes')
positive_set = {1, 2, 3, 4}
negative_set = {5, 6, 7, 8}
POS_num = 0
NEG_num = 0
SE_num = 0
result = {}
total_POS_num = 0
total_NEG_num = 0
total_SE_num = 0
exit_POS_num = 0
exit_NEG_num = 0
exit_SE_num = 0
exit_POS_num_all = 0
exit_NEG_num_all = 0
exit_SE_num_all = 0
exit_POS_num_4 = 0
exit_NEG_num_4 = 0
exit_SE_num_4 = 0
exit_POS_num_3 = 0
exit_NEG_num_3 = 0
exit_SE_num_3 = 0
exit_POS_num_2 = 0
exit_NEG_num_2 = 0
exit_SE_num_2 = 0
count = {}
for data in datalist:
    id = data['image_id']
    attr = data['attr_gt']
    # list(itertools.chain(*arr))
    attr_pos = attr.get(0, -1)
    attr_neg = attr.get(1, -1)
    attr_else = attr.get(2, -1)
    positive = []
    negative = []
    somethinge_else = []
    total_num = len(data['emo_label'])
    emo_label = data['emo_label']
    tokens_ids = data['tokens_ids']
    count[id] = {'POS': [], 'NEG': [], 'SE': []}
    for i in range(total_num):
        if emo_label[i] in positive_set:
            positive.append(tokens_ids[i])
        elif emo_label[i] in negative_set:
            negative.append(tokens_ids[i])
        elif emo_label[i] == 0:
            somethinge_else.append(tokens_ids[i])
    count_dic = {'POS': 0, 'NEG': 0, 'SE': 0, 'POS_ALL': 0, 'NEG_ALL': 0, 'SE_ALL': 0, 'POS_4': 0, 'NEG_4': 0,
                 'SE_4': 0, 'POS_3':0, 'NEG_3':0, 'SE_3':0, 'POS_2':0, 'NEG_2': 0, 'SE_2': 0}
    for sent in positive:
        if isinstance(attr_pos, np.ndarray):
            check = np.isin(sent, attr_pos)
            # 检查sent token与attr token之间是否存在相同的值
            # check_all = np.all(np.isin(attr_pos, sent))
            # 检查attr是否全部命中
            num_true = np.count_nonzero(check)
            num_false = sent.size - num_true
            # 检查命中占比
            intersect = np.intersect1d(sent, attr_pos)
            # 输出attr与sent的交集，判断命中次数
            if intersect.size >= 5:
                count_dic['POS_ALL'] += 1
                count_dic['POS_4'] += 1
                count_dic['POS_3'] += 1
                count_dic['POS_2'] += 1
            elif intersect.size >= 4:
                count_dic['POS_4'] += 1
                count_dic['POS_3'] += 1
                count_dic['POS_2'] += 1
            elif intersect.size >= 3:
                count_dic['POS_3'] += 1
                count_dic['POS_2'] += 1
            elif intersect.size >= 2:
                count_dic['POS_2'] += 1
            str = "{}:{}".format(num_true, num_false)
            count[id]['POS'].append(str)
            if num_true:
                count_dic['POS'] += 1
            # if check_all:
            #     count_dic['POS_ALL'] += 1
    for sent in negative:
        if isinstance(attr_neg, np.ndarray):
            check = np.isin(sent, attr_neg)
            # check_all = np.all(np.isin(attr_neg, sent))
            num_true = np.count_nonzero(check)
            num_false = sent.size - num_true
            intersect = np.intersect1d(sent, attr_neg)
            if intersect.size >= 5:
                count_dic['NEG_ALL'] += 1
                count_dic['NEG_4'] += 1
                count_dic['NEG_3'] += 1
                count_dic['NEG_2'] += 1
            if intersect.size >= 4:
                count_dic['NEG_4'] += 1
                count_dic['NEG_3'] += 1
                count_dic['NEG_2'] += 1
            elif intersect.size >= 3:
                count_dic['NEG_3'] += 1
                count_dic['NEG_2'] += 1
            elif intersect.size >= 2:
                count_dic['NEG_2'] += 1
            str = "{}:{}".format(num_true, num_false)
            count[id]['NEG'].append(str)
            if num_true:
                count_dic['NEG'] += 1
            # if check_all:
            #     count_dic['NEG_ALL'] += 1
    for sent in somethinge_else:
        if isinstance(attr_else, np.ndarray):
            check = np.isin(sent, attr_else)
            # check_all = np.all(np.isin(attr_else, sent))
            num_true = np.count_nonzero(check)
            num_false = sent.size - num_true
            intersect = np.intersect1d(sent, attr_else)
            if intersect.size >= 5:
                count_dic['SE_ALL'] += 1
                count_dic['SE_4'] += 1
                count_dic['SE_3'] += 1
                count_dic['SE_2'] += 1
            elif intersect.size >= 4:
                count_dic['SE_4'] += 1
                count_dic['SE_3'] += 1
                count_dic['SE_2'] += 1
            elif intersect.size >= 3:
                count_dic['SE_3'] += 1
                count_dic['SE_2'] += 1
            elif intersect.size >= 2:
                count_dic['SE_2'] += 1
            str = "{}:{}".format(num_true, num_false)
            count[id]['SE'].append(str)
            if num_true:
                count_dic['SE'] += 1
            # if check_all:
            #     count_dic['SE_ALL'] += 1
    result[id] = ("POS:{0}/{1}, NEG:{2}/{3}, SE:{4}/{5}".format(count_dic['POS'], len(positive), count_dic['NEG'],
                                                                len(negative), count_dic['SE'], len(somethinge_else)))
    total_POS_num += len(positive)
    total_NEG_num += len(negative)
    total_SE_num += len(somethinge_else)
    exit_POS_num += count_dic['POS']
    exit_NEG_num += count_dic['NEG']
    exit_SE_num += count_dic['SE']
    exit_POS_num_all += count_dic['POS_ALL']
    exit_NEG_num_all += count_dic['NEG_ALL']
    exit_SE_num_all += count_dic['SE_ALL']
    exit_POS_num_4 += count_dic['POS_4']
    exit_NEG_num_4 += count_dic['NEG_4']
    exit_SE_num_4 += count_dic['SE_4']
    exit_POS_num_3 += count_dic['POS_3']
    exit_NEG_num_3 += count_dic['NEG_3']
    exit_SE_num_3 += count_dic['SE_3']
    exit_POS_num_2 += count_dic['POS_2']
    exit_NEG_num_2 += count_dic['NEG_2']
    exit_SE_num_2 += count_dic['SE_2']


print("POS:{0}/{1}, NEG:{2}/{3}, SE:{4}/{5}".format(exit_POS_num, total_POS_num, exit_NEG_num, total_NEG_num,
                                                    exit_SE_num, total_SE_num))
print("5 hit:POS:{0}/{1}, NEG:{2}/{3}, SE:{4}/{5}".format(exit_POS_num_all, total_POS_num, exit_NEG_num_all,
                                                          total_NEG_num, exit_SE_num_all, total_SE_num))
print("4 hit:POS:{0}/{1}, NEG:{2}/{3}, SE:{4}/{5}".format(exit_POS_num_4, total_POS_num, exit_NEG_num_4,
                                                          total_NEG_num, exit_SE_num_4, total_SE_num))
print("3 hit:POS:{0}/{1}, NEG:{2}/{3}, SE:{4}/{5}".format(exit_POS_num_3, total_POS_num, exit_NEG_num_3,
                                                          total_NEG_num, exit_SE_num_3, total_SE_num))
print("2 hit:POS:{0}/{1}, NEG:{2}/{3}, SE:{4}/{5}".format(exit_POS_num_2, total_POS_num, exit_NEG_num_2,
                                                          total_NEG_num, exit_SE_num_2, total_SE_num))

with open("/home/wyf/codes/xmodaler base/xmodaler 3.13/tools/final_result.json", 'w') as f:
    json.dump(count, f)

with open("/home/wyf/codes/xmodaler base/xmodaler 3.13/tools/final_result2.json", 'w') as f:
    json.dump(count, f)


