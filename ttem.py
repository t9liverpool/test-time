from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import math
import util
import sys
import numpy as np
import time
import torch.nn.functional as F
import copy
import time
import sklearn
from torch.nn import DataParallel
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, roc_auc_score
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
import data_preprocess

from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import comb
from scipy.stats import norm

import queue
import json

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import discriminators
import scipy.stats as stats


def my_tta(opt, plain_model, prior_threshold, replay_size_classwise, weak_level,
            feature_stream, label_stream,
            test_known_feature, test_known_label,
            test_unknown_feature, test_unknown_label,
            replay_known_feature_classwise):

    memory_module = LONG_SHORT_TERM_MEM(plain_model, prior_threshold, test_known_feature, test_unknown_feature, opt, feature_stream, label_stream)


    sample_id = 0
    stride_size = opt.stride_size

    while sample_id < len(label_stream):


        print("进度：", sample_id, len(label_stream), "样本标签：",
              label_stream[sample_id:sample_id+10])


        memory_module.batch_online(feature_stream[sample_id:sample_id+stride_size])


        with torch.no_grad():


            select_id = memory_module.batch_regress(feature_stream[sample_id:sample_id + stride_size])
            select_id = [i for i in range(len(select_id)) if select_id[i]==True]

            # select_id = [i for i in range(len(select_id))]

            if len(select_id)>0:

                inputv = feature_stream[sample_id:sample_id + stride_size][select_id]

                if memory_module.filter_buff == None:
                    memory_module.filter_buff = inputv
                else:
                    memory_module.filter_buff = torch.cat([memory_module.filter_buff, inputv],dim=0)

                while len(memory_module.filter_buff)>=opt.negative_num:


                    discriminator = discriminators.GENERAL_GAUSSIAN_DISCRIMINATOR(opt,
                                                                                  replay_known_feature_classwise,
                                                                                  test_known_feature,
                                                                                  test_unknown_feature)


                    discriminator.batch_train(memory_module.filter_buff[:opt.negative_num])
                    discriminator.test()

                    memory_module.register_short_memory(discriminator)

                    memory_module.filter_buff = memory_module.filter_buff[opt.speed:]



        sample_id = sample_id + stride_size

        ########################注册长期记忆
        if sample_id % opt.stride_size == 0:

            memory_module.register_long_memory()
            # memory_module.final_test()
            # if input()=="x":
            #     exit()
            if sample_id % 1280 == 0:
                memory_module.final_test()
                # if input()=="x":
                #     exit()





class LONG_SHORT_TERM_MEM():
    def __init__(self, plain_model, prior_threshold, test_known_feature, test_unknown_feature, opt, feature_stream, label_stream):

        self.feature_stream = feature_stream
        self.label_stream = label_stream


        self.opt = opt

        self.plain_model = plain_model
        self.prior_threshold = prior_threshold

        self.test_unknown_feature = test_unknown_feature
        self.test_known_feature = test_known_feature

        self.short_term_memory = []
        self.long_term_memory = []

        self.short_term_memory_size = opt.short_term_memory_size
        self.long_term_memory_size = opt.long_term_memory_size
        self.env_sensitivity = opt.env_sensitivity

        self.detected = []
        self.miss_detected = []
        self.old_detected = []
        self.old_miss_detected = []


        self.bingo_record = []
        self.baseline_bingo_record = []
        self.bayes_bingo_record = []
        self.executive_memory = []
        self.key_point = [0.95]

        self.filter_buff = None


        for i in range(len(self.key_point)):
            self.bingo_record.append([])
            self.bayes_bingo_record.append([])
            self.baseline_bingo_record.append([])
            self.executive_memory.append([])


    def register_short_memory(self, model):

        while len(self.short_term_memory) >= self.opt.short_term_memory_size:

            if self.larmack_short_memory_disuse()==-1:
                break

        self.short_term_memory.append(model)


    def larmack_short_memory_disuse(self):

        disuse_candidate = -1
        least_score = 10000

        for i in range(len(self.short_term_memory)):

            # print(i, sum(self.short_term_memory[i].unique_activate_record), len(self.short_term_memory[i].unique_activate_record))

            if len(self.short_term_memory[i].unique_activate_record)<self.opt.short_term_memory_size-20:
                continue

            usage_score = sum(self.short_term_memory[i].unique_activate_record)/len(self.short_term_memory[i].unique_activate_record)

            if usage_score < least_score:
                least_score = usage_score
                disuse_candidate = i


        if disuse_candidate==-1:
            return -1

        del self.short_term_memory[disuse_candidate]

    def register_long_memory(self):

        if len(self.long_term_memory)==self.long_term_memory_size+1:

            self.larmack_long_memory_disuse()


        best_usage_score = -1
        best_id = -1

        for i in range(len(self.short_term_memory) - 1):

            usage_score = sum(self.short_term_memory[i].unique_activate_record)/(len(self.short_term_memory[i].unique_activate_record)+1)


            ######  debug 代码
            # if usage_score > best_usage_score:
            #
            #         print(i,  "    新遇见",
            #               len(self.short_term_memory[i].activate_record), "新独立激活",
            #               sum(self.short_term_memory[i].unique_activate_record), "评分：", usage_score,)
            #
            #         print("测试全类检出：%.2f%%， 测试全类召回%.2f%%" % (
            #         (len(self.short_term_memory[i].test_detected) / len(
            #             self.short_term_memory[i].test_unknown_feature) * 100),
            #         (len(self.short_term_memory[i].test_false_detected) / len(
            #             self.short_term_memory[i].test_known_feature) * 100)))
            ######  debug 代码

            if (usage_score > best_usage_score) and (
                    sum(self.short_term_memory[
                            i].unique_activate_record) >0) and (len(self.short_term_memory[i].activate_record)>49):
                best_usage_score = usage_score
                best_id = i

        if best_id == -1:
            if self.opt.debug_mode:
                print("本轮没有合格的边界")
            return
        if self.opt.debug_mode:
            print("选择第%d个边界id，新增边界编号 %d" % (len(self.long_term_memory) + 1, best_id))




        # 添加长期记忆
        # self.long_term_memory.append(copy.deepcopy(self.short_term_memory[best_id]))
        self.long_term_memory.append(copy.copy(self.short_term_memory[best_id]))

        #补充测试
        self.long_term_memory[-1].test()
        #更新执行空间
        self.executive_memory_update()




        ##  更新短期记忆unique记录

        for kk in range(len(self.short_term_memory)):
            self.short_term_memory[kk].unique_activate_record = []

        for kk in range(len(self.short_term_memory)):

            search_length = len(self.short_term_memory[kk].activate_record)

            for kkkk in range(len(self.long_term_memory)):
                if len(self.long_term_memory[kkkk].activate_record) < search_length:
                    search_length = len(self.long_term_memory[kkkk].activate_record)

            for kkk in range(1, 1 + search_length):
                if len(self.short_term_memory[kk].activate_record) < kkk:
                    pass
                else:
                    false_flag = False

                    for jj in range(len(self.long_term_memory)):

                        if len(self.long_term_memory[jj].activate_record) < kkk:
                            pass

                        elif self.long_term_memory[jj].activate_record[-kkk] == True:
                            false_flag = True
                            break

                    if false_flag:
                        self.short_term_memory[kk].unique_activate_record.append(False)
                    else:
                        self.short_term_memory[kk].unique_activate_record.append(
                            self.short_term_memory[kk].activate_record[-kkk])


        # 删除晋级短期记忆
        del self.short_term_memory[best_id]

        ###补充测试
        if self.opt.debug_mode:
            # self.old_detected = copy.deepcopy(self.detected)
            # self.old_miss_detected = copy.deepcopy(self.miss_detected)


            self.old_detected = copy.copy(self.detected)
            self.old_miss_detected = copy.copy(self.miss_detected)


            for ijk in range(len(self.test_unknown_feature)):
                  if self.long_term_memory[-1].predict_in_batch(self.long_term_memory[-1].test_unknown_feature[ijk])[0] == False:
                    self.detected.append(ijk)

            self.detected = list(set(self.detected))

            for ijk in range(len(self.test_known_feature)):
                if self.long_term_memory[-1].predict_in_batch(self.test_known_feature[ijk])[0] == False:
                    self.miss_detected.append(ijk)

            self.miss_detected = list(set(self.miss_detected))


        if self.opt.debug_mode:
            self.test_after_update()




    def larmack_long_memory_disuse(self):


        disuse_id = -1
        least_usage = 100000000
        # if self.opt.debug_mode:
        #     print("下面显示每个长期记忆的激活率是多少：")
        for i in range(len(self.long_term_memory)):
            # if self.opt.debug_mode:
            #     print("第",i, "个长期记忆体的激活率：",sum(self.long_term_memory[i].activate_record) / len(self.long_term_memory[i].activate_record))
            #     print("第", i, "个长期记忆体独立激活率：",
            #           sum(self.long_term_memory[i].unique_activate_record) / len(self.long_term_memory[i].unique_activate_record))
            #
            #     print("第", i, "个长期记忆体独立激活细则：",sum(self.long_term_memory[i].unique_activate_record), len(self.long_term_memory[i].unique_activate_record) )


            if sum(self.long_term_memory[i].unique_activate_record) / len(self.long_term_memory[i].unique_activate_record)<least_usage:
                least_usage = sum(self.long_term_memory[i].unique_activate_record) / len(self.long_term_memory[i].unique_activate_record)
                disuse_id = i
            # if self.opt.debug_mode:
            #     print("==============")

        if self.opt.debug_mode:
            print("删除长期记忆：", disuse_id)
        # 删除长期记忆
        self.long_term_memory = [self.long_term_memory[i] for i in range(len(self.long_term_memory)) if i != disuse_id]


        ##  更新短期记忆unique记录

        for kk in range(len(self.short_term_memory)):
            self.short_term_memory[kk].unique_activate_record = []

        for kk in range(len(self.short_term_memory)):

            search_length = len(self.short_term_memory[kk].activate_record)

            for kkkk in range(len(self.long_term_memory)):
                if len(self.long_term_memory[kkkk].activate_record) < search_length:
                    search_length = len(self.long_term_memory[kkkk].activate_record)

            for kkk in range(1, 1 + search_length):
                if len(self.short_term_memory[kk].activate_record) < kkk:
                    pass
                else:
                    false_flag = False

                    for jj in range(len(self.long_term_memory)):

                        if len(self.long_term_memory[jj].activate_record) < kkk:
                            pass

                        elif self.long_term_memory[jj].activate_record[-kkk] == True:
                            false_flag = True
                            break

                    if false_flag:
                        self.short_term_memory[kk].unique_activate_record.append(False)
                    else:
                        self.short_term_memory[kk].unique_activate_record.append(
                            self.short_term_memory[kk].activate_record[-kkk])







    def batch_online(self, features):

        long_term_memory_prediction_matrix = None
        short_term_memory_prediction_matrix = None


        for i in range(len(self.long_term_memory)):

            preds = self.long_term_memory[i].predict_in_batch(features)
            preds = torch.tensor(preds)

            if long_term_memory_prediction_matrix == None:
                long_term_memory_prediction_matrix = torch.unsqueeze(preds, dim=0)
            else:
                long_term_memory_prediction_matrix = torch.cat([long_term_memory_prediction_matrix, torch.unsqueeze(preds, dim=0)], dim=0)

        for i in range(len(self.short_term_memory)):

            # print("这里每次算一次判别器？")
            # print(self.short_term_memory[i].predict_in_batch(features, virtual_label))
            # print(self.short_term_memory[i].predict_in_batch(features, virtual_label))

            preds = self.short_term_memory[i].predict_in_batch(features)
            preds = torch.tensor(preds)

            if short_term_memory_prediction_matrix == None:
                short_term_memory_prediction_matrix = torch.unsqueeze(preds, dim=0)
            else:
                short_term_memory_prediction_matrix = torch.cat([short_term_memory_prediction_matrix, torch.unsqueeze(preds, dim=0)], dim=0)

        if len(self.long_term_memory) != 0:

            # long_term_memory_activate_matrix = copy.deepcopy(long_term_memory_prediction_matrix)
            long_term_memory_activate_matrix = copy.copy(long_term_memory_prediction_matrix)
            long_term_memory_activate_matrix = (~long_term_memory_activate_matrix)*1

            for i in range(long_term_memory_activate_matrix.shape[0]):
                if i == 0:
                    continue
                if i != 0:
                    long_term_memory_activate_matrix[i,:] = long_term_memory_activate_matrix[i,:] + long_term_memory_activate_matrix[i-1,:]

        ##### 更新长期个体独立激活
        for i in range(len(self.long_term_memory)):

            extend_details = [((long_term_memory_prediction_matrix[i][j]==False) and (long_term_memory_activate_matrix[i][j]==1)).item() for j in range(len(long_term_memory_activate_matrix[i]))]

            self.long_term_memory[i].unique_activate_record.extend(extend_details)


        ##### 更新长期个体一般激活
        for i in range(len(self.long_term_memory)):

            self.long_term_memory[i].activate_record.extend([(not long_term_memory_prediction_matrix[i][j].item()) for j in range(len(long_term_memory_prediction_matrix[i]))])

        # ##### 更新短期个体独立激活和一般激活
        for i in range(len(self.short_term_memory)):

            self.short_term_memory[i].activate_record.extend([(not short_term_memory_prediction_matrix[i][j].item()) for j in range(len(short_term_memory_prediction_matrix[i]))])

            if len(self.long_term_memory) != 0:

                extend_details = [((short_term_memory_prediction_matrix[i][j]==False) and (long_term_memory_activate_matrix[-1][j]==0)).item() for j in range(len(short_term_memory_prediction_matrix[i]))]
                self.short_term_memory[i].unique_activate_record.extend(extend_details)
            else:
                self.short_term_memory[i].unique_activate_record.extend([(not short_term_memory_prediction_matrix[i][j].item()) for j in range(len(short_term_memory_prediction_matrix[i]))])



    def test_after_update(self):

        print("\n")
        print("总检出未知类：", len(self.detected), "未知类总数：",
              len(self.test_unknown_feature), "检出率：",
              len(self.detected) / len(self.test_unknown_feature), "新增检出：",
              len(self.detected) - len(self.old_detected))
        print("\n")

        # print(detected)

        print("总误检已知类：", len(self.miss_detected), "已知类总数：",
              len(self.test_known_feature), "1-误检率：",
              1 - len(self.miss_detected) / len(self.test_known_feature), "新增误检：",
              len(self.miss_detected) - len(self.old_miss_detected))


    def final_test(self):

        ##### 这个测试其实是近似，按照的是用进废退，并非严格的贪心算法。

        key_point = self.key_point

        long_term_memory_pool = self.long_term_memory

        executive_long_term_memory = []
        false_detected_buff = []

        detected = []
        miss_detected = []
        miss_detected2 = []

        for tpr in key_point:
            print("###################################     TPR:",tpr, "      ###################################")

            ### 每次增加一个最棒的判别器
            for step in range(len(self.long_term_memory)):

                print("每次增加一个最棒的判别器：", step)

                best_usage = -1
                selected_id = -1

                for i in range(len(long_term_memory_pool)):

                    false_detected_buff_this = long_term_memory_pool[i].test_false_detected

                    # print(i, len(list(set(false_detected_buff)|set(false_detected_buff_this))), len(self.test_known_feature), len(list(set(false_detected_buff)|set(false_detected_buff_this)))/len(self.test_known_feature), tpr)
                    if 1-len(list(set(false_detected_buff)|set(false_detected_buff_this)))/(len(self.test_known_feature)//2)<tpr:
                        continue


                    elif sum(long_term_memory_pool[i].unique_activate_record) / len(
                            long_term_memory_pool[i].unique_activate_record) > best_usage:

                        best_usage = sum(long_term_memory_pool[i].unique_activate_record) / len(
                            long_term_memory_pool[i].unique_activate_record)

                        selected_id = i

                if selected_id == -1:
                    break


                false_detected_buff.extend(long_term_memory_pool[selected_id].test_false_detected)
                false_detected_buff = list(set(false_detected_buff))
                # print("选中：",selected_id, "当前正确率：", 1-len(false_detected_buff)/len(self.test_known_feature))

                executive_long_term_memory.append(long_term_memory_pool[selected_id])
                long_term_memory_pool = [long_term_memory_pool[id] for id in range(len(long_term_memory_pool)) if id != selected_id]

                detected.extend(executive_long_term_memory[-1].test_detected)
                detected = list(set(detected))

                miss_detected.extend(executive_long_term_memory[-1].test_false_detected)
                miss_detected = list(set(miss_detected))

                miss_detected2.extend(executive_long_term_memory[-1].test_false_detected2)
                miss_detected2 = list(set(miss_detected2))


                print("总检出未知类：", len(detected), "未知类总数：",
                      len(self.test_unknown_feature), "检出率：",
                      len(detected) / len(self.test_unknown_feature), "新增检出：",
                      0)

                print("验证集总误检已知类：", len(miss_detected), "验证集已知类总数：",
                      len(self.test_known_feature)//2, "验证集1-误检率：",
                      1 - len(miss_detected) / (len(self.test_known_feature)//2), "验证集新增误检：",
                      0)

                print("测试集总误检已知类：", len(miss_detected2), "测试集已知类总数：",
                      len(self.test_known_feature)//2, "测试集1-误检率：",
                      1 - len(miss_detected2) / (len(self.test_known_feature)//2), "测试集新增误检：",
                      0)



    def executive_memory_update(self):

        key_point = self.key_point

        long_term_memory_pool = self.long_term_memory
        false_detected_buff = []
        self.executive_memory = []

        for i in range(len(key_point)):
            self.executive_memory.append([])


        for index in range(len(key_point)):

            tpr = key_point[index]

            ### 每次增加一个最棒的判别器
            for step in range(len(self.long_term_memory)):

                best_usage = -1
                selected_id = -1

                for i in range(len(long_term_memory_pool)):

                    false_detected_buff_this = long_term_memory_pool[i].test_false_detected

                    # print(i, len(list(set(false_detected_buff)|set(false_detected_buff_this))), len(self.test_known_feature), len(list(set(false_detected_buff)|set(false_detected_buff_this)))/len(self.test_known_feature), tpr)
                    if 1 - len(list(set(false_detected_buff) | set(false_detected_buff_this))) / len(
                            self.test_known_feature) < tpr:
                        continue


                    elif sum(long_term_memory_pool[i].unique_activate_record) / len(
                            long_term_memory_pool[i].unique_activate_record) > best_usage:

                        best_usage = sum(long_term_memory_pool[i].unique_activate_record) / len(
                            long_term_memory_pool[i].unique_activate_record)

                        selected_id = i

                if selected_id == -1:
                    break

                false_detected_buff.extend(long_term_memory_pool[selected_id].test_false_detected)
                false_detected_buff = list(set(false_detected_buff))
                # print("选中：",selected_id, "当前正确率：", 1-len(false_detected_buff)/len(self.test_known_feature))

                self.executive_memory[index].append(long_term_memory_pool[selected_id])
                long_term_memory_pool = [long_term_memory_pool[id] for id in range(len(long_term_memory_pool)) if
                                         id != selected_id]



    def batch_regress(self, features):

        if features.dim() == 1:
            features = torch.unsqueeze(features, dim=0)

        executive_memory_prediction_matrix_list = []

        for discriminators_id in range(len(self.executive_memory)):

            executive_memory_prediction_matrix = None

            for discriminator in self.executive_memory[discriminators_id]:

                preds = discriminator.predict_in_batch(features)
                preds = torch.tensor(preds)
                if executive_memory_prediction_matrix == None:
                    executive_memory_prediction_matrix = torch.unsqueeze(preds, dim=0)
                else:
                    executive_memory_prediction_matrix = torch.cat(
                        [executive_memory_prediction_matrix, torch.unsqueeze(preds, dim=0)], dim=0)

            if executive_memory_prediction_matrix!=None:
                executive_memory_prediction_matrix_list.append(executive_memory_prediction_matrix)


        executive_memory_activate_matrix_list = []
        for discriminators_id in range(len(executive_memory_prediction_matrix_list)):

            # executive_memory_activate_matrix = copy.deepcopy(executive_memory_prediction_matrix_list[discriminators_id])
            executive_memory_activate_matrix = copy.copy(executive_memory_prediction_matrix_list[discriminators_id])
            executive_memory_activate_matrix = (~executive_memory_activate_matrix) * 1

            for i in range(executive_memory_activate_matrix.shape[0]):
                if i == 0:
                    continue
                if i != 0:
                    executive_memory_activate_matrix[i, :] = executive_memory_activate_matrix[i,
                                                             :] + executive_memory_activate_matrix[i - 1, :]

            executive_memory_activate_matrix_list.append(executive_memory_activate_matrix)


        for discriminators_id in range(len(executive_memory_activate_matrix_list)):

            if discriminators_id>0:

                executive_memory_activate_matrix_list[discriminators_id][-1] += executive_memory_activate_matrix_list[discriminators_id-1][-1]



        for discriminators_id in range(len(self.executive_memory)):

            if len(executive_memory_activate_matrix_list)==0:
                self.bingo_record[discriminators_id].extend([False for i in range(len(features))])
            elif discriminators_id+1<=len(executive_memory_activate_matrix_list):
                self.bingo_record[discriminators_id].extend([(executive_memory_activate_matrix_list[discriminators_id][-1][i]>0).item() for i in range(len(features))])
            else:
                self.bingo_record[discriminators_id].extend(self.bingo_record[discriminators_id-1][-len(features):])




        for record_id in range(len(self.baseline_bingo_record)):

            self.plain_model = self.plain_model.to(self.opt.device)

            if self.prior_threshold[-1] >= 1:
                score, _ = torch.max(
                    self.plain_model(features.to(self.opt.device), features=True),
                    dim=1)

                self.plain_model = self.plain_model.cpu()

                print("使用logit先验")

            else:
                score, _ = torch.max(torch.nn.functional.softmax(self.plain_model(features.to(self.opt.device), features = True), dim=1), dim=1)
                print("使用softmax先验")


            extend_details = [(score[i] < self.prior_threshold[self.opt.prior_strength]).item() for i in range(len(score))]

            self.baseline_bingo_record[record_id].extend(extend_details)




        for record_id in range(len(self.bayes_bingo_record)):

            print( "先验激活程度:",sum(self.baseline_bingo_record[record_id][-self.env_sensitivity:]), "后验激活程度:",sum(self.bingo_record[record_id][-self.env_sensitivity:]) )

            if sum(self.baseline_bingo_record[record_id][-self.env_sensitivity:])>sum(self.bingo_record[record_id][-self.env_sensitivity:]):

                self.bayes_bingo_record[record_id].extend(self.baseline_bingo_record[record_id][-len(features):])

                print("敏感度：", self.env_sensitivity, "使用先验函数决策")
                return self.baseline_bingo_record[record_id][-len(features):]
            else:
                self.bayes_bingo_record[record_id].append(self.bingo_record[record_id][-len(features):])

                print("敏感度：", self.env_sensitivity, "使用后验函数决策")
                return self.bingo_record[record_id][-len(features):]

