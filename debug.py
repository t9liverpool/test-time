##########################################################################################
#
#
#      TTA开放集算法项目
#
#
#########################################################################################


####常用公开库库
from __future__ import print_function
import argparse
import os
import random
import torch

print("========== ENV ==========")
print("Torch:", torch.__version__)
print("Torch path:", torch.__file__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("==========================")

import torch.nn as nn
import torch.autograd as autograd
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import math
import sys
import numpy as np
import time
import torch.nn.functional as F
from sklearn.cluster import KMeans
import copy
import time
import sklearn
from torch.nn import DataParallel
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, roc_auc_score
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from progress.bar import Bar as Bar
from scipy.special import comb
from scipy.stats import norm
import queue
import json
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn import svm


from util import compute_oscr, FeatDataset, compute_roc, AverageMeter
import data_preprocess

import softmax_baseline
import clip
import ttem

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default=None)
parser.add_argument('--split', type=int, default=None, help='split id')
args = parser.parse_args()


class RUN:

    def __init__(self):

        global args

        general_set()

        if args.baseline_type == "softmax":
            prefix = "plain_softmax_"
        if args.baseline_type == "clip":
            prefix = "clip_softmax_"

        if os.path.exists("./checkpoints/debug_receive_data_" + prefix + args.dataset + "_" + str(args.split) + '.pth'):
            # if False:
            print("读取./checkpoints/debug_receive_data_" + prefix + args.dataset + "_" + str(args.split) + '.pth')

            self.data_dict = torch.load(
                "./checkpoints/debug_receive_data_" + prefix + args.dataset + "_" + str(args.split) + '.pth',
                map_location=args.device)

            args.data = data_preprocess.DATA_LOADER(args)

        else:
            args.data = data_preprocess.DATA_LOADER(args)
            self.split_train_replay_online_test(args.baseline_type)

        args.nclass = self.data_dict["nclass"]

        print("\n")
        print("################################   train softmax baseline  ################################")

        unknownset = FeatDataset(
            data=[self.data_dict["test_unknown_feature"][:2500], self.data_dict["test_unknown_label"][:2500]])
        testset = FeatDataset(
            data=[self.data_dict["test_known_feature"][:2500], self.data_dict["test_known_label"][:2500]])
        testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)
        unknownloader = DataLoader(unknownset, batch_size=128, shuffle=False, num_workers=0)

        self.plain_model = softmax_baseline.train(args.baseline_type, self.data_dict["input_dim"],
                                                  self.data_dict["nclass"], args,
                                                  self.data_dict["train_known_feature"],
                                                  self.data_dict["train_known_label"],
                                                  self.data_dict["test_known_feature"],
                                                  self.data_dict["test_known_label"],
                                                  self.data_dict["test_unknown_feature"],
                                                  self.data_dict["test_unknown_label"])

        with torch.no_grad():
            prior_threshold = softmax_baseline.val(True, self.plain_model, testloader, unknownloader, args)

        print("################################   train softmax Ends  ################################")
        print("\n")

        print("################################   开始构建在线数据集，数据灌入类型:", args.data_stream_type,
              " ################################")
        unknown_label_list = torch.tensor(list(set([i.item() for i in self.data_dict["online_unknown_label"]])))
        unknown_label_list = [i.item() for i in unknown_label_list]
        data_stream_setting = {}
        data_stream_setting["uniform"] = [unknown_label_list]
        data_stream_setting["seq"] = [[i] for i in unknown_label_list]

        print("./checkpoints/debug_datastream_" + prefix + args.dataset + "_" + str(args.split) + "_" + str(
            args.data_stream_type) + '.pth')
        print(os.path.exists(
            "./checkpoints/debug_datastream_" + prefix + args.dataset + "_" + str(args.split) + "_" + str(
                args.data_stream_type) + '.pth'))

        if os.path.exists("./checkpoints/debug_datastream_" + prefix + args.dataset + "_" + str(args.split) + "_" + str(
                args.data_stream_type) + '.pth'):
            # if False:
            print("读取./checkpoints/debug_datastream_" + prefix + args.dataset + "_" + str(args.split) + "_" + str(
                args.data_stream_type) + '.pth')

            datastream = torch.load(
                "./checkpoints/debug_datastream_" + prefix + args.dataset + "_" + str(args.split) + "_" + str(
                    args.data_stream_type) + '.pth', map_location=args.device)

            self.feature_stream = datastream["feature_stream"]
            self.label_stream = datastream["label_stream"]

        else:
            self.prepare_data_stream(args.baseline_type, data_stream_setting[args.data_stream_type])

        if args.feed_in_known == True:
            self.feed_in_known(args)

        print("已知类类别：", args.data.seenclasses)

        #######################有监督上限   ########################
        print("=======测试上限===========")
        self.test_supervised_counterpart()
        print("=======测试上限结束===========")
        # # ######################有监督上限   ########################


        ttem.my_tta(args, self.plain_model, prior_threshold,
                    args.replay_size_classwise, args.bias_level,
                    self.feature_stream, self.label_stream,
                    self.data_dict["test_known_feature"][:4000], self.data_dict["test_known_label"][:4000],
                    self.data_dict["test_unknown_feature"][:4000],
                    self.data_dict["test_unknown_label"][:4000],
                    self.data_dict["replay_known_feature_classwise"])
        print("################################   This is larmack2  ################################")
        # exit()

    def test_supervised_counterpart(self):

        train_size = 2000

        train_features = torch.cat(
            [self.data_dict["online_known_feature"][:train_size], self.data_dict["online_unknown_feature"][:train_size]],
            dim=0)

        train_labels = torch.cat(
            [torch.ones(len(self.data_dict["online_known_feature"][:train_size])),
             (-1) * torch.ones(len(self.data_dict["online_unknown_feature"][:train_size]))], dim=0)

        C = 10.0
        kernal_type = "rbf"

        discriminator = svm.SVC(kernel=kernal_type, C=C)
        discriminator.fit(train_features.cpu(), train_labels.cpu())

        test_features = torch.cat(
            [self.data_dict["test_known_feature"][:1000], self.data_dict["test_unknown_feature"][:1000]], dim=0)

        test_labels = torch.cat(
            [torch.ones(len(self.data_dict["test_known_feature"][:1000])),
             (-1) * torch.ones(len(self.data_dict["test_unknown_feature"][:1000]))],
            dim=0)

        print("二分类精度：", discriminator.score(test_features.cpu(), test_labels.cpu()))

        in_score = discriminator.decision_function(self.data_dict["test_known_feature"][:1000].cpu())
        out_score = discriminator.decision_function(self.data_dict["test_unknown_feature"][:1000].cpu())
        print("AUROC:", compute_roc(in_score, out_score))

        id_score_list = in_score
        ood_score_list = out_score

        score, index = torch.sort(torch.tensor(id_score_list))
        threshod = score[int(len(score) * 0.05)]

        tnr = torch.tensor([1 for i in range(len(ood_score_list)) if ood_score_list[i] < threshod])
        tnr = (torch.sum(tnr) / len(ood_score_list)).item()

        print("tnr:", tnr)

        tpr = torch.tensor([1 for i in range(len(id_score_list)) if id_score_list[i] >= threshod])
        tpr = (torch.sum(tpr) / len(id_score_list)).item()

        print("tpr:", tpr)

    def feed_in_known(self, args):

        print(len(self.data_dict["online_known_feature"]))
        # self.data_dict["online_known_feature"] = self.data_dict["online_known_feature"][:148]
        # self.data_dict["online_known_label"] = self.data_dict["online_known_label"][:148]
        print(len(self.data_dict["online_unknown_feature"]))
        print("注意调整比例！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！")
        # exit()


        if args.dataset !="imagenet1k":

            self.data_dict["online_known_feature"] = torch.cat([self.data_dict["online_known_feature"] for i in range(300)],
                                                               dim=0)
            self.data_dict["online_known_label"] = torch.cat([self.data_dict["online_known_label"] for i in range(300)],
                                                             dim=0)

        if len(self.feature_stream) * args.online_known_vs_unknown > len(self.data_dict["online_known_feature"]):

            self.feature_stream = self.feature_stream[
                                  :int(len(self.data_dict["online_known_feature"]) / args.online_known_vs_unknown)]
            self.label_stream = self.label_stream[
                                :int(len(self.data_dict["online_known_feature"]) / args.online_known_vs_unknown)]


        else:
            self.data_dict["online_known_feature"] = self.data_dict["online_known_feature"][
                                                     :int(len(self.feature_stream) * args.online_known_vs_unknown)]
            self.data_dict["online_known_label"] = self.data_dict["online_known_label"][
                                                   :int(len(self.feature_stream) * args.online_known_vs_unknown)]

        self.data_dict["online_unknown_feature"] = self.feature_stream

        if args.dataset in ["cifar10-svhn", "imagenet1k"]:
            self.label_stream = -self.label_stream
            self.data_dict["online_unknown_label"] = self.label_stream
        else:
            self.data_dict["online_unknown_label"] = self.label_stream

        self.feature_stream = torch.cat(
            [self.feature_stream.to(args.device), self.data_dict["online_known_feature"].to(args.device)], dim=0)
        self.label_stream = torch.cat(
            [self.label_stream.to(args.device), self.data_dict["online_known_label"].to(args.device)], dim=0)

        shuffle_index = torch.randperm(len(self.label_stream))
        self.feature_stream = self.feature_stream[shuffle_index]
        self.label_stream = self.label_stream[shuffle_index]

        print("在线集已知类：", len(self.data_dict["online_known_feature"]), "在线集未知类：",
              len(self.data_dict["online_unknown_feature"]))
        print("测试集已知类：", len(self.data_dict["test_known_feature"]), "测试集未知类：",
              len(self.data_dict["test_unknown_feature"]))

    def prepare_data_stream(self, baseline_type, data_stream_setting):

        feature_stream = None
        label_stream = None

        splits_per_unknown_label = {}
        samples_per_unknown_split = {}
        unknown_label_set = set(
            [self.data_dict["online_unknown_label"][i].int().item() for i in
             range(len(self.data_dict["online_unknown_label"]))])
        for label in unknown_label_set:
            splits_per_unknown_label[label] = 0

        for group_labels in data_stream_setting:
            for label in group_labels:
                splits_per_unknown_label[label] += 1

        for label in unknown_label_set:
            samples_per_unknown_split[label] = int(
                len(self.data_dict["online_unknown_feature_classwise"][label]) / splits_per_unknown_label[label])

        for i in range(len(data_stream_setting)):

            feature_stream_buff = None
            label_stream_buff = None

            for j in range(len(data_stream_setting[i])):
                if feature_stream_buff == None:
                    feature_stream_buff = self.data_dict["online_unknown_feature_classwise"][data_stream_setting[i][j]][
                                          :samples_per_unknown_split[data_stream_setting[i][j]]]
                    label_stream_buff = torch.ones(len(
                        self.data_dict["online_unknown_feature_classwise"][data_stream_setting[i][j]][
                        :samples_per_unknown_split[data_stream_setting[i][j]]])) * data_stream_setting[i][j]

                    self.data_dict["online_unknown_feature_classwise"][data_stream_setting[i][j]] = \
                    self.data_dict["online_unknown_feature_classwise"][data_stream_setting[i][j]][
                    samples_per_unknown_split[data_stream_setting[i][j]]:]

                else:
                    feature_stream_buff = torch.cat([feature_stream_buff,
                                                     self.data_dict["online_unknown_feature_classwise"][
                                                         data_stream_setting[i][j]][
                                                     :samples_per_unknown_split[data_stream_setting[i][j]]]])
                    label_stream_buff = torch.cat(
                        [label_stream_buff, torch.ones(len(
                            self.data_dict["online_unknown_feature_classwise"][data_stream_setting[i][j]][
                            :samples_per_unknown_split[data_stream_setting[i][j]]])) * data_stream_setting[i][j]])

                    self.data_dict["online_unknown_feature_classwise"][data_stream_setting[i][j]] = \
                        self.data_dict["online_unknown_feature_classwise"][data_stream_setting[i][j]][
                        samples_per_unknown_split[data_stream_setting[i][j]]:]

            shuffle_index = torch.randperm(len(label_stream_buff))
            feature_stream_buff = feature_stream_buff[shuffle_index]
            label_stream_buff = label_stream_buff[shuffle_index]

            if feature_stream == None:
                feature_stream = feature_stream_buff
                label_stream = label_stream_buff
            else:
                feature_stream = torch.cat([feature_stream, feature_stream_buff])
                label_stream = torch.cat([label_stream, label_stream_buff])

        self.feature_stream = feature_stream
        self.label_stream = label_stream

        datastream = {}
        datastream["feature_stream"] = feature_stream
        datastream["label_stream"] = label_stream

        if baseline_type == "softmax":
            prefix = "plain_softmax_"
        if baseline_type == "clip":
            prefix = "clip_softmax_"

        torch.save(datastream,
                   "./checkpoints/debug_datastream_" + prefix + args.dataset + "_" + str(args.split) + "_" + str(
                       args.data_stream_type) + '.pth')

    def split_train_replay_online_test(self, baseline_type):

        global args
        self.device = args.device
        self.data = args.data
        self.nclass = self.data.seenclasses
        self.input_dim = self.data.train_known_feature.shape[1]

        self.data_dict = {}

        self.data_dict["train_known_feature"], self.data_dict[
            "train_known_label"] = self.data.train_known_feature, self.data.train_known_label

        self.data_dict["test_known_feature"], self.data_dict[
            "test_known_label"] = self.data.test_known_feature, self.data.test_known_label

        self.data_dict["train_unknown_feature"], self.data_dict[
            "train_unknown_label"] = self.data.train_unknown_feature, self.data.train_unknown_label

        self.data_dict["test_unknown_feature"], self.data_dict[
            "test_unknown_label"] = self.data.test_unknown_feature, self.data.test_unknown_label

        print("\n")
        print("################################   data report  ################################")
        print("接收数据: 已知训练，已知测试，未知训练，未知测试", len(self.data_dict["train_known_feature"]),
              len(self.data_dict["test_known_feature"]), len(self.data_dict["train_unknown_feature"]),
              len(self.data_dict["test_unknown_feature"]))

        ##########################做一些class-wise变量

        if args.debug_mode:
            print("已知类训练集字典")
        self.data_dict["train_known_feature_classwise"] = {}
        known_label_set = set([self.data_dict["train_known_label"][i].int().item() for i in
                               range(len(self.data_dict["train_known_label"]))])

        for select_class in known_label_set:
            index = [i for i in range(len(self.data_dict["train_known_label"])) if
                     self.data_dict["train_known_label"][i] == select_class]
            train_known_feature = self.data_dict["train_known_feature"][index]

            self.data_dict["train_known_feature_classwise"][select_class] = train_known_feature

        if args.debug_mode:
            print("未知类训练集字典")
        self.data_dict["online_unknown_feature_classwise"] = {}
        unknown_label_set = set([self.data_dict["train_unknown_label"][i].int().item() for i in
                                 range(len(self.data_dict["train_unknown_label"]))])

        for select_class in unknown_label_set:
            self.data_dict["online_unknown_feature_classwise"][select_class] = []

        for i in range(len(self.data_dict["train_unknown_label"])):
            self.data_dict["online_unknown_feature_classwise"][self.data_dict["train_unknown_label"][i].item()].append(
                torch.unsqueeze(self.data_dict["train_unknown_feature"][i], dim=0))

        for key in self.data_dict["online_unknown_feature_classwise"]:
            self.data_dict["online_unknown_feature_classwise"][key] = torch.cat(
                self.data_dict["online_unknown_feature_classwise"][key], dim=0)

        ###############################################

        if args.debug_mode:
            print("已知类测试集字典")
        self.data_dict["test_known_feature_classwise "] = {}
        known_label_set = set([self.data_dict["test_known_label"][i].int().item() for i in
                               range(len(self.data_dict["test_known_label"]))])

        for select_class in known_label_set:
            index = [i for i in range(len(self.data_dict["test_known_label"])) if
                     self.data_dict["test_known_label"][i] == select_class]
            test_known_feature = self.data_dict["test_known_feature"][index]

            self.data_dict["test_known_feature_classwise "][select_class] = test_known_feature

        ###### imagenet1K设置成长尾

        if args.debug_mode:
            print("未知类测试集字典")
        self.data_dict["test_unknown_feature_classwise "] = {}
        unknown_label_set = set([self.data_dict["test_unknown_label"][i].int().item() for i in
                                 range(len(self.data_dict["test_unknown_label"]))])

        for select_class in unknown_label_set:
            index = [i for i in range(len(self.data_dict["test_unknown_label"])) if
                     self.data_dict["test_unknown_label"][i] == select_class]
            test_unknown_feature = self.data_dict["test_unknown_feature"][index]

            self.data_dict["test_unknown_feature_classwise "][select_class] = test_unknown_feature

        self.data_dict["replay_known_feature_classwise"] = {}
        self.data_dict["online_known_feature_classwise"] = {}


        if args.dataset not in ["CUB", "AwA", "aPaY", "SUN", "imagenet1k"] and (args.baseline_type!="clip"):
            ########################### 划分已知类在线数据集 ##################
            # 已知类的在线集测试集比例

            for class_id in self.data_dict["test_known_feature_classwise "].keys():
                total_sample = len(self.data_dict["test_known_feature_classwise "][class_id])

                self.data_dict["replay_known_feature_classwise"][class_id] = \
                self.data_dict["test_known_feature_classwise "][class_id][
                :int(total_sample / 3)]

                self.data_dict["online_known_feature_classwise"][class_id] = \
                self.data_dict["test_known_feature_classwise "][class_id][
                int(total_sample / 3):int(total_sample / 3) * 2]

                self.data_dict["test_known_feature_classwise "][class_id] = \
                self.data_dict["test_known_feature_classwise "][class_id][
                int(total_sample / 3) * 2:]

                # self.data_dict["replay_known_feature_classwise"][class_id] = torch.cat(
                #     [self.data_dict["replay_known_feature_classwise"][class_id],
                #      self.data_dict["train_known_feature_classwise"][class_id]], dim=0)


        else:

            for class_id in self.data_dict["test_known_feature_classwise "].keys():
                total_sample = len(self.data_dict["test_known_feature_classwise "][class_id]) + len(
                    self.data_dict["train_known_feature_classwise"][class_id])

                sample_pool = torch.cat([self.data_dict["test_known_feature_classwise "][class_id],
                                         self.data_dict["train_known_feature_classwise"][class_id]], dim=0)

                self.data_dict["online_known_feature_classwise"][class_id] = sample_pool[
                                                                             :int(total_sample / 4)]

                self.data_dict["replay_known_feature_classwise"][class_id] = sample_pool[
                                                                             int(total_sample / 4):int(
                                                                                 total_sample / 2)].to(args.device)

                self.data_dict["test_known_feature_classwise "][class_id] = sample_pool[
                                                                            int(total_sample / 2):int(
                                                                                total_sample / 4 * 3)]

                self.data_dict["train_known_feature_classwise"][class_id] = sample_pool[
                                                                            int(total_sample / 4 * 3):]

                self.data_dict["replay_known_feature_classwise"][class_id] = torch.cat([self.data_dict["replay_known_feature_classwise"][class_id],
                                         self.data_dict["train_known_feature_classwise"][class_id]], dim=0)

        ### 重组 非class-wise数据   replay就不设置非class-wise了

        self.data_dict["train_known_feature"] = torch.cat(
            [self.data_dict["train_known_feature_classwise"][class_id] for class_id in
             self.data_dict["train_known_feature_classwise"].keys()], dim=0)
        self.data_dict["train_known_label"] = torch.cat(
            [(torch.ones(len(self.data_dict["train_known_feature_classwise"][class_id])) * class_id).long() for class_id
             in
             self.data_dict["train_known_feature_classwise"].keys()], dim=0)

        self.data_dict["online_known_feature"] = torch.cat(
            [self.data_dict["online_known_feature_classwise"][class_id] for class_id in
             self.data_dict["online_known_feature_classwise"].keys()],
            dim=0)
        self.data_dict["online_known_label"] = torch.cat(
            [(torch.ones(len(self.data_dict["online_known_feature_classwise"][class_id])) * class_id).long() for
             class_id in
             self.data_dict["online_known_feature_classwise"].keys()], dim=0)

        self.data_dict["online_unknown_feature"] = torch.cat(
            [self.data_dict["online_unknown_feature_classwise"][class_id] for class_id in
             self.data_dict["online_unknown_feature_classwise"].keys()],
            dim=0)
        self.data_dict["online_unknown_label"] = torch.cat(
            [(torch.ones(len(self.data_dict["online_unknown_feature_classwise"][class_id])) * class_id).long() for
             class_id in
             self.data_dict["online_unknown_feature_classwise"].keys()], dim=0)

        self.data_dict["test_known_feature"] = torch.cat(
            [self.data_dict["test_known_feature_classwise "][class_id] for class_id in
             self.data_dict["test_known_feature_classwise "].keys()], dim=0)
        self.data_dict["test_known_label"] = torch.cat(
            [(torch.ones(len(self.data_dict["test_known_feature_classwise "][class_id])) * class_id).long() for class_id
             in self.data_dict["test_known_feature_classwise "].keys()], dim=0)

        self.data_dict["test_unknown_feature"] = torch.cat(
            [self.data_dict["test_unknown_feature_classwise "][class_id] for class_id in
             self.data_dict["test_unknown_feature_classwise "].keys()],
            dim=0)
        self.data_dict["test_unknown_label"] = torch.cat(
            [(torch.ones(len(self.data_dict["test_unknown_feature_classwise "][class_id])) * class_id).long() for
             class_id in
             self.data_dict["test_unknown_feature_classwise "].keys()], dim=0)

        ####### shuffle  #######
        shuffle_index = torch.randperm(len(self.data_dict["train_known_label"]))
        self.data_dict["train_known_feature"] = self.data_dict["train_known_feature"][shuffle_index]
        self.data_dict["train_known_label"] = self.data_dict["train_known_label"][shuffle_index]

        shuffle_index = torch.randperm(len(self.data_dict["online_known_label"]))
        self.data_dict["online_known_feature"] = self.data_dict["online_known_feature"][shuffle_index]
        self.data_dict["online_known_label"] = self.data_dict["online_known_label"][shuffle_index]

        shuffle_index = torch.randperm(len(self.data_dict["online_unknown_label"]))
        self.data_dict["online_unknown_feature"] = self.data_dict["online_unknown_feature"][shuffle_index]
        self.data_dict["online_unknown_label"] = self.data_dict["online_unknown_label"][shuffle_index]

        shuffle_index = torch.randperm(len(self.data_dict["test_known_label"]))
        self.data_dict["test_known_feature"] = self.data_dict["test_known_feature"][shuffle_index]
        self.data_dict["test_known_label"] = self.data_dict["test_known_label"][shuffle_index]

        shuffle_index = torch.randperm(len(self.data_dict["test_unknown_label"]))
        self.data_dict["test_unknown_feature"] = self.data_dict["test_unknown_feature"][shuffle_index]
        self.data_dict["test_unknown_label"] = self.data_dict["test_unknown_label"][shuffle_index]

        print("已知闭集训练：", len(self.data_dict["train_known_feature"]),"已知在线：", len(self.data_dict["online_known_feature"]),"未知在线：", len(self.data_dict["online_unknown_feature"]),
              "已知测试：", len(self.data_dict["test_known_feature"]),"未知测试：", len(self.data_dict["test_unknown_feature"]))

        ####### shuffle  #######

        self.data_dict["device"] = args.device
        self.data_dict["nclass"] = self.data.seenclasses
        self.data_dict["input_dim"] = self.data.train_known_feature.shape[1]

        if baseline_type == "softmax":
            prefix = "plain_softmax_"
        if baseline_type == "clip":
            prefix = "clip_softmax_"

        torch.save(self.data_dict,
                   "./checkpoints/debug_receive_data_" + prefix + args.dataset + "_" + str(args.split) + '.pth')

        print("整理数据：训练已知类样本数量", len(self.data_dict["train_known_feature"]), "在线已知类样本数量",
              len(self.data_dict["online_known_feature"]),
              "测试已知类样本数量", len(self.data_dict["test_known_feature"]), "在线未知类样本数量",
              len(self.data_dict["online_unknown_feature"]),
              "测试未知类样本数量", len(self.data_dict["test_unknown_feature"]))
        print("################################   data report  ################################")
        print("\n")


def compute_roc(known_scores, unknown_scores):
    y_true = np.array([1] * len(known_scores) + [0] * len(unknown_scores))
    y_score = np.concatenate([known_scores, unknown_scores])
    # fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)
    return auc_score


def general_set():
    global args

    args.debug_mode = True
    # args.debug_mode = False

    # args.dataset = "cifar-10-10"
    # args.dataset = "cifar10-svhn"
    # args.dataset = "CUB"
    # args.dataset = "AwA"
    # args.dataset = "aPaY"
    # args.dataset = "SUN"
    # args.dataset = "cifar100"
    # args.dataset = "tinyimagenet"
    args.dataset = "imagenet1k"
    # args.dataset = "svhn-cifar10"
    # args.dataset = "cifar100"
    # args.dataset = "svhn"

    args.data_stream_type = "seq"
    args.data_stream_type = "uniform"

    args.feed_in_known = True
    # args.feed_in_known = False
    args.online_known_vs_unknown = 0.0001
    args.online_known_vs_unknown = 1
    # args.online_known_vs_unknown = 0.2

    args.long_term_memory_size = 500
    args.short_term_memory_size = 500
    args.optimization_cycle = 100
    args.env_sensitivity = 128


    args.stride_size = 64
    args.prior_strength = 9
    args.speed = 1

    if args.dataset == "svhn":
        args.baseline_type = "softmax"
        # args.baseline_type = "clip"
        args.replay_size_classwise = 100
        args.bias_level = 0.002
        args.prior_strength = 2
        args.negative_num = int(10 * args.replay_size_classwise / 2 / 6)

    if args.dataset == "aPaY":
        args.replay_size_classwise = 100
        args.baseline_type = "softmax"
        args.bias_level = 0.0005
        args.prior_strength = 5
        args.negative_num = int(6 * args.replay_size_classwise / 12)

    if args.dataset == "AwA":
        args.replay_size_classwise = 100
        args.baseline_type = "softmax"
        args.bias_level = 0.0005
        args.prior_strength = 5
        args.negative_num = int(6 * args.replay_size_classwise / 12 )

    if args.dataset == "CUB":
        args.replay_size_classwise = 30
        args.baseline_type = "softmax"
        args.bias_level = 0.001
        args.prior_strength = 0
        args.negative_num = int(20 * args.replay_size_classwise / 12)

    if args.dataset == "cifar100":
        args.replay_size_classwise = 100
        # args.baseline_type = "clip"
        args.baseline_type = "softmax"
        args.bias_level = 0.001
        args.prior_strength = 0
        args.negative_num = int(6 * args.replay_size_classwise / 12)

    if args.dataset == "cifar-10-10":
        args.replay_size_classwise = 100
        args.baseline_type = "clip"
        # args.baseline_type = "softmax"
        args.bias_level = 0.001
        args.prior_strength = 9
        args.negative_num = int(6 * args.replay_size_classwise / 12)

        # args.replay_size_classwise = 150

    if args.dataset == "cifar10-svhn":
        args.baseline_type = "softmax"
        # args.baseline_type = "clip"
        args.replay_size_classwise = 100
        args.bias_level = 0.001
        args.prior_strength = 9
        args.negative_num = int(10 * args.replay_size_classwise / 2 / 6)

    if args.dataset == "svhn-cifar10":
        args.baseline_type = "softmax"
        # args.baseline_type = "clip"
        args.replay_size_classwise = 100
        args.bias_level = 0.004
        args.prior_strength = 2
        args.negative_num = int(10 * args.replay_size_classwise / 2 / 6)

    if args.dataset == "imagenet1k":
        args.baseline_type = "clip"
        args.replay_size_classwise = 10
        args.bias_level = 0.001
        args.prior_strength = 9
        args.negative_num = int(500 * args.replay_size_classwise / 12)
        args.speed = 20

    if args.dataset == "tinyimagenet":
        args.replay_size_classwise = 100
        args.baseline_type = "softmax"
        args.baseline_type = "clip"
        args.bias_level = 0.01
        args.prior_strength = 0
        args.negative_num = int(20 * args.replay_size_classwise/ 12)



    args.dataroot = "../data/xlsa17"
    args.image_embedding = "res101"
    args.class_embedding = 'att'

    args.manualSeed = random.randint(1, 10000)

    args.manualSeed = 888

    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    torch.cuda.manual_seed_all(args.manualSeed)
    cudnn.benchmark = True
    torch.backends.cudnn.enabled = False
    # args.device = torch.device("cuda:"+str(int(args.gpu)) if torch.cuda.is_available() else "cpu")
    args.device = torch.device("cuda:" + str(1) if torch.cuda.is_available() else "cpu")
    # args.device = torch.device("cpu")

    if args.split == None:
        args.split = 0
        args.split = 1
        args.split = 2
        args.split = 3
        # args.split = 4

    print("Random Seed: ", args.manualSeed)


if __name__ == '__main__':
    obj = RUN()