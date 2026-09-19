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
import json
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import psutil



class GENERAL_GAUSSIAN_DISCRIMINATOR():

    train_known_feature = None
    validate_known_feature = None
    test_known_feature = None
    test_known_feature2 = None
    test_unknown_feature = None
    replay_known_feature_classwise = None





    def __init__(self, opt, replay_known_feature_classwise, test_known_feature,test_unknown_feature,):

        self.discriminator = None
        self.activate_record = []
        self.unique_activate_record = []
        self.test_false_detected = None
        self.test_false_detected2 = None
        self.test_detected = None



        if GENERAL_GAUSSIAN_DISCRIMINATOR.test_known_feature == None:

            GENERAL_GAUSSIAN_DISCRIMINATOR.opt = opt


            GENERAL_GAUSSIAN_DISCRIMINATOR.replay_known_feature_classwise = replay_known_feature_classwise
            GENERAL_GAUSSIAN_DISCRIMINATOR.test_known_feature = test_known_feature[:len(test_known_feature)//2]
            GENERAL_GAUSSIAN_DISCRIMINATOR.test_known_feature2 = test_known_feature[len(test_known_feature)//2:]
            GENERAL_GAUSSIAN_DISCRIMINATOR.test_unknown_feature = test_unknown_feature

            GENERAL_GAUSSIAN_DISCRIMINATOR.replay_known_feature = torch.cat([replay_known_feature_classwise[this_key] for this_key in replay_known_feature_classwise.keys()], dim=0)
            shuffle_index = torch.randperm(len(GENERAL_GAUSSIAN_DISCRIMINATOR.replay_known_feature))
            GENERAL_GAUSSIAN_DISCRIMINATOR.replay_known_feature = GENERAL_GAUSSIAN_DISCRIMINATOR.replay_known_feature[shuffle_index]

            GENERAL_GAUSSIAN_DISCRIMINATOR.replay_known_feature = GENERAL_GAUSSIAN_DISCRIMINATOR.replay_known_feature[:opt.replay_size_classwise*opt.nclass]
            split_num = int(len(GENERAL_GAUSSIAN_DISCRIMINATOR.replay_known_feature) / 2)


            GENERAL_GAUSSIAN_DISCRIMINATOR.validate_known_feature = GENERAL_GAUSSIAN_DISCRIMINATOR.replay_known_feature[split_num:]
            GENERAL_GAUSSIAN_DISCRIMINATOR.replay_known_feature = GENERAL_GAUSSIAN_DISCRIMINATOR.replay_known_feature[:split_num]



        GENERAL_GAUSSIAN_DISCRIMINATOR.threshold = opt.bias_level



    def batch_train(self, online_feature):

        replay_known_feature = GENERAL_GAUSSIAN_DISCRIMINATOR.replay_known_feature.cpu()

        train_features = torch.cat([replay_known_feature.cpu(), online_feature.cpu()], dim=0)

        # print("用于训练的已知类样本数量和未知类样本数量", len(replay_known_feature), len(online_feature))
        # exit()

        train_labels = torch.cat(
            [torch.ones(len(replay_known_feature)), (-1) * torch.ones(len(online_feature))], dim=0).cpu()

        C = 100.0
        kernal_type = "rbf"

        self.discriminator = svm.SVC(kernel=kernal_type, C=C)
        self.discriminator.fit(train_features.cpu(), train_labels.cpu())

        # print("训练集大小：", len(replay_known_feature), len(online_feature))
        # print("svm训练结果：",
        #       self.discriminator.score(train_features.cpu(), train_labels.cpu()))
        # print("svm测试检出率：",
        #       self.discriminator.score(self.test_unknown_feature.cpu(), -1*torch.ones(len(self.test_unknown_feature.cpu()))))
        # print("svm测试召回率：",
        #       self.discriminator.score(self.test_known_feature.cpu(), torch.ones(len(self.test_known_feature.cpu()))))

        # if input()=="x":
        #     exit()



        validate_known_feature = GENERAL_GAUSSIAN_DISCRIMINATOR.validate_known_feature.cpu()


        map_known_data = torch.tensor(self.discriminator.decision_function(validate_known_feature.cpu()))

        # print("验证集数量：", len(validate_known_feature))


        self.mean = map_known_data.mean()
        self.std_dev = map_known_data.std()

        # print("均值：", self.mean)
        # print("方差：", self.std_dev)

        from torch.distributions import Normal
        self.dist = Normal(loc=self.mean, scale=self.std_dev)




    def test(self):

        map_data = torch.tensor(self.discriminator.decision_function(
            GENERAL_GAUSSIAN_DISCRIMINATOR.test_known_feature.cpu()))

        prob = self.dist.cdf(map_data)


        pred = [(int(1) * (((prob[i] > 0.5) ) or (
                (prob[i] <= 0.5) and (prob[i] > (GENERAL_GAUSSIAN_DISCRIMINATOR.threshold / 2))))).item() for i in range(len(prob))]

        self.test_false_detected = [i
                                                      for i in range(len(pred)) if
                                                      pred[i] == 0]


        # print("测试召回率", len(self.test_false_detected)/len(pred))

        # class -wise 检测率

        map_data = torch.tensor(self.discriminator.decision_function(
            GENERAL_GAUSSIAN_DISCRIMINATOR.test_unknown_feature.cpu()))

        prob = self.dist.cdf(map_data)


        pred = [(int(1) * (((prob[i] > 0.5)) or (
                (prob[i] <= 0.5) and (prob[i] > (GENERAL_GAUSSIAN_DISCRIMINATOR.threshold / 2))))).item() for i in range(len(prob))]

        self.test_detected = [i for
                                                i in range(len(pred)) if
                                                pred[i] == 0]



        ##############################################

        # print("测试检出率", len(self.test_detected) / len(pred))

        map_data = torch.tensor(self.discriminator.decision_function(
            GENERAL_GAUSSIAN_DISCRIMINATOR.test_known_feature2.cpu()))

        prob = self.dist.cdf(map_data)

        pred = [(int(1) * (((prob[i] > 0.5)) or (
                (prob[i] <= 0.5) and (prob[i] > (GENERAL_GAUSSIAN_DISCRIMINATOR.threshold / 2))))).item() for i in
                range(len(prob))]

        self.test_false_detected2 = [i
                                    for i in range(len(pred)) if
                                    pred[i] == 0]

        # print("测试召回率", len(self.test_false_detected2)/len(pred))
        #
        #
        # if input()=="x":
        #     exit()




    def predict_in_batch(self, feature):
        #### 已知类预测True，未知类预测False
        ####  非本类，全部认知为已知类。


        #####  下面是使用分布做预测
        if feature.dim() == 1:
            feature = torch.unsqueeze(feature, dim=0)

        map_data = torch.tensor(self.discriminator.decision_function(
            feature.cpu()))
        prob = self.dist.cdf(map_data)


        pred = [(int(1) * (((prob[i] > 0.5)) or (
                (prob[i] <= 0.5) and (prob[i] > (GENERAL_GAUSSIAN_DISCRIMINATOR.threshold / 2))))).item() for i in range(len(prob))]




        pred = [True if (pred[i] - 1) == 0 else False for i in range(len(pred))]

        return pred

