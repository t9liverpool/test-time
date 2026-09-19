#import h5py
import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
import sys
from sklearn.cluster import KMeans
import os
import random

#### 读取数据集。
# 一共四个数据集，已知训练，已知测试。未知训练，未知测试。但是比例需要整清楚。
# 注意已知类和未知类的标签，都要处理到0-N的标签中去。

class DATA_LOADER(object):
    def __init__(self, opt):


        if (opt.dataset == "cifar100"):
            self.read_cifar100(opt)

        if (opt.dataset == "cifar10-svhn"):
            self.read_cifar10_svhn(opt)
        if (opt.dataset == "svhn-cifar10"):
            self.read_svhn_cifar10(opt)

        if (opt.dataset == "cifar-10-10"):
            self.read_cifar10(opt)
        if (opt.dataset == "cifar-10-100"):
            self.read_cifar10_100(opt)
        if opt.dataset == "CUB":
            self.read_matdataset_CUB(opt)
        if opt.dataset == "AwA":
            self.read_matdataset_AwA(opt)
        if opt.dataset == "aPaY":
            self.read_matdataset_aPaY(opt)
        if opt.dataset == "SUN":
            self.read_matdataset_SUN(opt)
        if opt.dataset == "tinyimagenet":
            self.read_matdataset_tinyimagenet(opt)
        if opt.dataset == "imagenet1k":
            self.read_matdataset_imagenet1k(opt)

        if opt.dataset == "svhn":
            self.read_matdataset_svhn(opt)

        self.feature_dim = self.train_known_feature.shape[1]


    def read_matdataset_svhn(self, opt):
        if opt.baseline_type=="softmax":
            fileware = "softmax"
            prefix = "Softmax_"

        if opt.baseline_type=="clip":
            fileware = "clip_model"
            prefix = "CLIP_"

        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
        else:
            print("???")

        SAVE_PARA_PATH = "deepfeature/"+fileware+"/results/"+prefix+"svhn_split"+str(opt.split)+"_train_known_set.pth"
        self.train_known_feature, self.train_known_label = torch.load(SAVE_PARA_PATH)

        SAVE_PARA_PATH = "deepfeature/"+fileware+"/results/" + prefix+"svhn_split" + str(opt.split) + "_test_known_set.pth"
        self.test_known_feature, self.test_known_label = torch.load(SAVE_PARA_PATH)

        SAVE_PARA_PATH = "deepfeature/"+fileware+"/results/" + prefix+"svhn_split" + str(opt.split) + "_train_unknown_set.pth"
        self.train_unknown_feature, self.train_unknown_label = torch.load(SAVE_PARA_PATH)

        SAVE_PARA_PATH = "deepfeature/"+fileware+"/results/" + prefix+"svhn_split" + str(opt.split) + "_test_unknown_set.pth"
        self.test_unknown_feature, self.test_unknown_label = torch.load(SAVE_PARA_PATH)

        self.train_known_feature = self.train_known_feature.float()
        self.test_known_feature = self.test_known_feature.float()
        self.train_unknown_feature = self.train_unknown_feature.float()
        self.test_unknown_feature = self.test_unknown_feature.float()


        self.nclass = 6
        self.seenclasses = 6
        self.unseenclasses = 4





    def read_matdataset_imagenet1k(self, opt):

        if opt.baseline_type=="softmax":
            fileware = "softmax"
            prefix = "Softmax_"

        if opt.baseline_type=="clip":
            fileware = "clip_model"
            prefix = "CLIP_"

        SAVE_PARA_PATH = "deepfeature/"+fileware+"/results/"+prefix+"imagenet1k_split"+str(opt.split)+"_train_known_set.pth"
        self.train_known_feature, self.train_known_label = torch.load(SAVE_PARA_PATH)

        SAVE_PARA_PATH = "deepfeature/"+fileware+"/results/" + prefix+"imagenet1k_split" + str(opt.split) + "_test_known_set.pth"
        self.test_known_feature, self.test_known_label = torch.load(SAVE_PARA_PATH)

        SAVE_PARA_PATH = "deepfeature/"+fileware+"/results/" + prefix+"imagenet1k_split" + str(opt.split) + "_train_unknown_set.pth"
        self.train_unknown_feature, self.train_unknown_label = torch.load(SAVE_PARA_PATH)

        SAVE_PARA_PATH = "deepfeature/"+fileware+"/results/" + prefix+"imagenet1k_split" + str(opt.split) + "_test_unknown_set.pth"
        self.test_unknown_feature, self.test_unknown_label = torch.load(SAVE_PARA_PATH)

        self.train_known_feature = self.train_known_feature.float()
        self.test_known_feature = self.test_known_feature.float()
        self.train_unknown_feature = self.train_unknown_feature.float()
        self.test_unknown_feature = self.test_unknown_feature.float()


        self.nclass = 1000
        self.seenclasses = 500

    def read_cifar100(self, opt):
        if opt.baseline_type=="softmax":
            fileware = "softmax"
            prefix = "Softmax_"

        if opt.baseline_type=="clip":
            fileware = "clip_model"
            prefix = "CLIP_"

        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
        else:
            print("???")

        SAVE_PARA_PATH = "deepfeature/"+fileware+"/results/"+prefix+"cifar100_split"+str(opt.split)+"_train_known_set.pth"
        self.train_known_feature, self.train_known_label = torch.load(SAVE_PARA_PATH)

        SAVE_PARA_PATH = "deepfeature/"+fileware+"/results/" + prefix+"cifar100_split" + str(opt.split) + "_test_known_set.pth"
        self.test_known_feature, self.test_known_label = torch.load(SAVE_PARA_PATH)

        SAVE_PARA_PATH = "deepfeature/"+fileware+"/results/" + prefix+"cifar100_split" + str(opt.split) + "_train_unknown_set.pth"
        self.train_unknown_feature, self.train_unknown_label = torch.load(SAVE_PARA_PATH)

        SAVE_PARA_PATH = "deepfeature/"+fileware+"/results/" + prefix+"cifar100_split" + str(opt.split) + "_test_unknown_set.pth"
        self.test_unknown_feature, self.test_unknown_label = torch.load(SAVE_PARA_PATH)

        self.train_known_feature = self.train_known_feature.float()
        self.test_known_feature = self.test_known_feature.float()
        self.train_unknown_feature = self.train_unknown_feature.float()
        self.test_unknown_feature = self.test_unknown_feature.float()


        self.nclass = 10
        self.seenclasses = 10
        self.unseenclasses = 100


    def read_svhn_cifar10(self, opt):
        if opt.baseline_type=="softmax":
            fileware = "softmax"
            prefix = "Softmax_"

        if opt.baseline_type=="clip":
            fileware = "clip_model"
            prefix = "CLIP_"

        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
        else:
            print("???")

        SAVE_PARA_PATH = "deepfeature/"+fileware+"/results/"+prefix+"svhn-cifar10_split"+str(opt.split)+"_train_known_set.pth"
        self.train_known_feature, self.train_known_label = torch.load(SAVE_PARA_PATH)

        SAVE_PARA_PATH = "deepfeature/"+fileware+"/results/" + prefix+"svhn-cifar10_split" + str(opt.split) + "_test_known_set.pth"
        self.test_known_feature, self.test_known_label = torch.load(SAVE_PARA_PATH)

        SAVE_PARA_PATH = "deepfeature/"+fileware+"/results/" + prefix+"svhn-cifar10_split" + str(opt.split) + "_train_unknown_set.pth"
        self.train_unknown_feature, self.train_unknown_label = torch.load(SAVE_PARA_PATH)

        SAVE_PARA_PATH = "deepfeature/"+fileware+"/results/" + prefix+"svhn-cifar10_split" + str(opt.split) + "_test_unknown_set.pth"
        self.test_unknown_feature, self.test_unknown_label = torch.load(SAVE_PARA_PATH)

        self.train_known_feature = self.train_known_feature.float()
        self.test_known_feature = self.test_known_feature.float()
        self.train_unknown_feature = self.train_unknown_feature.float()
        self.test_unknown_feature = self.test_unknown_feature.float()


        self.nclass = 10
        self.seenclasses = 10
        self.unseenclasses = 100




    def read_cifar10_svhn(self, opt):

        if opt.baseline_type=="softmax":
            fileware = "softmax"
            prefix = "Softmax_"

        if opt.baseline_type=="clip":
            fileware = "clip_model"
            prefix = "CLIP_"

        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
        else:
            print("???")

        SAVE_PARA_PATH = "deepfeature/"+fileware+"/results/"+prefix+"cifar10-svhn_split"+str(opt.split)+"_train_known_set.pth"
        self.train_known_feature, self.train_known_label = torch.load(SAVE_PARA_PATH)

        SAVE_PARA_PATH = "deepfeature/"+fileware+"/results/" + prefix+"cifar10-svhn_split" + str(opt.split) + "_test_known_set.pth"
        self.test_known_feature, self.test_known_label = torch.load(SAVE_PARA_PATH)

        SAVE_PARA_PATH = "deepfeature/"+fileware+"/results/" + prefix+"cifar10-svhn_split" + str(opt.split) + "_train_unknown_set.pth"
        self.train_unknown_feature, self.train_unknown_label = torch.load(SAVE_PARA_PATH)

        SAVE_PARA_PATH = "deepfeature/"+fileware+"/results/" + prefix+"cifar10-svhn_split" + str(opt.split) + "_test_unknown_set.pth"
        self.test_unknown_feature, self.test_unknown_label = torch.load(SAVE_PARA_PATH)

        self.train_known_feature = self.train_known_feature.float()
        self.test_known_feature = self.test_known_feature.float()
        self.train_unknown_feature = self.train_unknown_feature.float()
        self.test_unknown_feature = self.test_unknown_feature.float()


        self.nclass = 10
        self.seenclasses = 10
        self.unseenclasses = 100


    def read_cifar10_100(self, opt):



        if opt.baseline_type=="softmax":
            fileware = "softmax"
            prefix = "Softmax_"

        if opt.baseline_type=="clip":
            fileware = "clip_model"
            prefix = "CLIP_"

        SAVE_PARA_PATH = "deepfeature/"+fileware+"/results/"+prefix+"cifar-10-100_split"+str(opt.split)+"_train_known_set.pth"
        self.train_known_feature, self.train_known_label = torch.load(SAVE_PARA_PATH)

        SAVE_PARA_PATH = "deepfeature/"+fileware+"/results/" + prefix+"cifar-10-100_split" + str(opt.split) + "_test_known_set.pth"
        self.test_known_feature, self.test_known_label = torch.load(SAVE_PARA_PATH)

        SAVE_PARA_PATH = "deepfeature/"+fileware+"/results/" + prefix+"cifar-10-100_split" + str(opt.split) + "_train_unknown_set.pth"
        self.train_unknown_feature, self.train_unknown_label = torch.load(SAVE_PARA_PATH)

        SAVE_PARA_PATH = "deepfeature/"+fileware+"/results/" + prefix+"cifar-10-100_split" + str(opt.split) + "_test_unknown_set.pth"
        self.test_unknown_feature, self.test_unknown_label = torch.load(SAVE_PARA_PATH)

        self.train_known_feature = self.train_known_feature.float()
        self.test_known_feature = self.test_known_feature.float()
        self.train_unknown_feature = self.train_unknown_feature.float()
        self.test_unknown_feature = self.test_unknown_feature.float()


        self.nclass = 10
        self.seenclasses = 10
        self.unseenclasses = 100


    def read_matdataset_AwA(self, opt):




        ###################  这个数据集非常不均衡  #################


        matcontent = sio.loadmat(opt.dataroot + "/" + "AWA2" + "/" + opt.image_embedding + ".mat")
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(opt.dataroot + "/" + "AWA2" + "/" + opt.class_embedding + "_splits.mat")

        self.all_class_num = 50
        self.all_att_num = 85




        #################################################################################################################
        if opt.split == 3:
            # split_known_class = [48, 35, 25, 8, 24, 33, 20, 7, 15, 12, 2, 21, 16, 5, 36, 3, 40, 43, 31, 23]  # split0
            split_known_class = [2, 20, 13, 16, 49, 6, 8, 25, 26, 7, 18, 15, 28, 39, 43, 44, 30, 40, 0, 14]
            split_known_class = [2, 20, 13, 16, 49, 6, 8, 25, 26, 7]
        # if opt.split == 1:
        #     split_known_class = [15, 11, 34, 32, 20, 37, 38, 18, 39, 0, 45, 35, 41, 25, 31, 46, 5, 12, 30, 33]  # split1
        # if opt.split == 2:
        #     split_known_class = [2, 20, 13, 16, 49, 6, 8, 25, 26, 7, 18, 15, 28, 39, 43, 44, 30, 40, 0, 14]  # split2
        # if opt.split == 3:
        #     split_known_class = [2, 20, 13, 16, 49, 6, 8, 25, 26, 7, 18, 15, 28, 39, 43, 44, 30, 40, 0, 14]  # split3
        # if opt.split == 4:
        #     split_known_class = [2, 20, 13, 16, 49, 6, 8, 25, 26, 7, 18, 15, 28, 39, 43, 44, 30, 40, 0, 14]  # split4



        self.split_unknown = [i for i in range(50) if i not in split_known_class]


        self._att_name = []
        self._class_name = []
        self._train_ids = []
        self._test_ids = []
        self._open_ids = []
        self._image_id_label = {}
        # self.split_unknown = [i for i in range(1, self.nclass + 1) if i not in self.split_known]

        awa_split_ratio = 4/5
        random.seed(6666)


        self.class_sample_check_dict={}

        for i in range(self.all_class_num):
            self.class_sample_check_dict[i] = []

        for i in range(len(label)):
            self.class_sample_check_dict[label[i]].append(i)

        for i in range(len(self.class_sample_check_dict)):
            random.shuffle(self.class_sample_check_dict[i])
            train_len = int(len(self.class_sample_check_dict[i])*awa_split_ratio)
            if i in split_known_class:
                self._train_ids.extend(self.class_sample_check_dict[i][:train_len])
                self._test_ids.extend(self.class_sample_check_dict[i][train_len:])
            else:
                self._open_ids.extend(self.class_sample_check_dict[i][train_len:])

        for kk in range(len(self._train_ids)):
            self._train_ids[kk] = int(self._train_ids[kk])

        self._train_ids = np.array(self._train_ids)

        for kk in range(len(self._test_ids)):
            self._test_ids[kk] = int(self._test_ids[kk])

        self._test_ids = np.array(self._test_ids)

        for kk in range(len(self._open_ids)):
            self._open_ids[kk] = int(self._open_ids[kk])

        self._open_ids = np.array(self._open_ids)

        trainval_loc = self._train_ids
        test_seen_loc = self._test_ids
        test_unseen_loc = self._open_ids


        #################################################################################################################

        scaler = preprocessing.MinMaxScaler()

        _train_feature = scaler.fit_transform(feature[trainval_loc])

        _test_seen_feature = scaler.transform(feature[test_seen_loc])

        _test_unseen_feature = scaler.transform(feature[test_unseen_loc])

        self.train_seen_feature = torch.from_numpy(_train_feature).float()
        mx = self.train_seen_feature.max()
        self.train_seen_feature.mul_(1 / mx)

        self.seenclasses_buff = list(np.unique(torch.from_numpy(label[trainval_loc]).long().numpy()))
        self.unseenclasses_buff = list(np.unique(torch.from_numpy(label[test_unseen_loc]).long().numpy()))

        self.train_seen_label = torch.tensor([self.seenclasses_buff.index(ele) for ele in list(label[trainval_loc])]).long()

        self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
        self.test_unseen_feature.mul_(1 / mx)
        self.test_unseen_label = torch.tensor(
            [self.unseenclasses_buff.index(ele) for ele in list(label[test_unseen_loc])]).long()

        self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
        self.test_seen_feature.mul_(1 / mx)
        self.test_seen_label = torch.tensor(
            [self.seenclasses_buff.index(ele) for ele in list(label[test_seen_loc])]).long()

        self.seenclasses = torch.from_numpy(np.unique(torch.from_numpy(label[trainval_loc]).long().numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(torch.from_numpy(label[test_unseen_loc]).long().numpy()))

        self.seenclasses = len(self.seenclasses)
        self.unseenclasses = len(self.unseenclasses)

        self.train_known_feature, self.train_known_label = self.train_seen_feature, self.train_seen_label
        self.test_known_feature, self.test_known_label = self.test_seen_feature, self.test_seen_label
        self.unknown_feature, self.unknown_label = self.test_unseen_feature, self.test_unseen_label


        #### shuffle #####

        shuffle_index = torch.randperm(len(self.train_known_feature))
        self.train_known_feature = self.train_known_feature[shuffle_index]
        self.train_known_label = self.train_known_label[shuffle_index]



        shuffle_index = torch.randperm(len(self.test_known_feature))
        self.test_known_feature = self.test_known_feature[shuffle_index]
        self.test_known_label = self.test_known_label[shuffle_index]

        shuffle_index = torch.randperm(len(self.unknown_feature))
        self.unknown_feature = self.unknown_feature[shuffle_index]
        self.unknown_label = self.unknown_label[shuffle_index]



        # label unknowns as negatives
        self.unknown_label = torch.tensor([-self.unknown_label[i] - 1 for i in range(len(self.unknown_label))])

        #######

        test_ratio = len(self.test_known_label)/(len(self.train_known_label)+len(self.test_known_label))

        self.test_unknown_feature = self.unknown_feature[:int(len(self.unknown_feature)*test_ratio)]
        self.test_unknown_label = self.unknown_label[:int(len(self.unknown_label) * test_ratio)]

        self.train_unknown_feature = self.unknown_feature[int(len(self.unknown_feature) * test_ratio):]
        self.train_unknown_label = self.unknown_label[int(len(self.unknown_label) * test_ratio):]



    def read_matdataset_CUB(self, opt):


        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")


        #################################################################################################################
        #这个数据集标签对不上，但是id是对上的。这里已知类别必须从1开始，属性从0开始

        if opt.split == 4:
            split_known_class = [144, 73, 84, 182, 24, 29, 10, 48, 75, 1, 112, 19, 183, 135, 101, 95, 192, 163, 98, 14]#split4

        if opt.split == 3:
            split_known_class = [42, 99, 184, 121, 113, 17, 93, 157, 173, 192, 27, 35, 36, 103, 56, 145, 80, 85, 47, 89]#split3

        if opt.split == 2:
            split_known_class = [48, 49, 20, 114, 78, 157, 29, 59, 199, 177, 58, 76, 101, 168, 63, 53, 36, 6, 12, 182]#split2

        if opt.split == 1:
            split_known_class = [124, 55, 146, 112, 157, 29, 12, 63, 136, 186, 13, 94, 189, 47, 41, 182, 154, 200, 52, 106]#split1


        if opt.split == 0:
            split_known_class = [20, 166, 1, 101, 23, 84, 70, 57, 29, 102, 155, 97, 48, 53, 183, 110, 142, 106, 122, 15] #split0

        # split_known_class = [ i for i in range(1, 201)]
        # random.shuffle(split_known_class)
        # split_known_class = split_known_class[:50]

        self.split_unknown = [i for i in range(1, 200 + 1) if i not in split_known_class]

        self.root = "../data/CUB_200_2011/"
        self.nclass = 200
        self.split_known = split_known_class
        self.classes_file = os.path.join(self.root, 'classes.txt')  # <class_id> <class_name>
        self.att_file = os.path.join(self.root, 'attributes.txt')
        self.classes2att_file = os.path.join(self.root, 'class_attribute_labels_continuous.txt')
        self.image_class_labels_file = os.path.join(self.root, 'image_class_labels.txt')  # <image_id> <class_id>
        self.image_attribute_labels_file = os.path.join(self.root, 'image_attribute_labels.txt')
        self.images_file = os.path.join(self.root, 'images.txt')  # <image_id> <image_name>
        self.train_test_split_file = os.path.join(self.root, 'train_test_split.txt')  # <image_id> <is_training_image>

        self._att_name = []
        self._class_name = []
        self._train_ids = []
        self._test_ids = []
        self._open_ids = []
        self._open_online_ids = []
        self._image_id_label = {}

        self._train_test_split()
        self._get_id_to_label()

        self._open_split()


        for kk in range(len(self._train_ids)):
            self._train_ids[kk] = int(self._train_ids[kk])

        self._train_ids = np.array(self._train_ids)

        for kk in range(len(self._test_ids)):
            self._test_ids[kk] = int(self._test_ids[kk])

        self._test_ids = np.array(self._test_ids)

        for kk in range(len(self._open_ids)):
            self._open_ids[kk] = int(self._open_ids[kk])

        self._open_ids = np.array(self._open_ids)

        for kk in range(len(self._open_online_ids)):
            self._open_online_ids[kk] = int(self._open_online_ids[kk])

        self._open_online_ids = np.array(self._open_online_ids)

        trainval_loc = self._train_ids - 1
        test_seen_loc = self._test_ids - 1
        test_unseen_loc = self._open_ids - 1
        train_unseen_loc = self._open_online_ids-1


        #################################################################################################################

        scaler = preprocessing.MinMaxScaler()

        _train_feature = scaler.fit_transform(feature[trainval_loc])
        _test_seen_feature = scaler.transform(feature[test_seen_loc])
        _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
        _train_unseen_feature = scaler.transform(feature[train_unseen_loc])



        self.train_feature = torch.from_numpy(_train_feature).float()
        mx = self.train_feature.max()
        self.train_feature.mul_(1/mx)


        self.seenclasses_buff = list(np.unique(torch.from_numpy(label[trainval_loc]).long().numpy()))
        self.unseenclasses_buff = list(np.unique(torch.from_numpy(label[test_unseen_loc]).long().numpy()))

        self.train_label = torch.tensor([self.seenclasses_buff.index(ele) for ele in list(label[trainval_loc])]).long()



        self.train_unseen_feature = torch.from_numpy(_train_unseen_feature).float()
        self.train_unseen_feature.mul_(1 / mx)
        self.train_unseen_label = torch.tensor(
            [self.unseenclasses_buff.index(ele) for ele in list(label[train_unseen_loc])]).long()


        self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
        self.test_unseen_feature.mul_(1/mx)
        self.test_unseen_label = torch.tensor([self.unseenclasses_buff.index(ele) for ele in list(label[test_unseen_loc])]).long()


        self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
        self.test_seen_feature.mul_(1/mx)
        self.test_seen_label = torch.tensor([self.seenclasses_buff.index(ele) for ele in list(label[test_seen_loc])]).long()




        self.seenclasses = torch.from_numpy(np.unique(torch.from_numpy(label[trainval_loc]).long().numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(torch.from_numpy(label[test_unseen_loc]).long().numpy()))

        self.seenclasses = len(self.seenclasses)
        self.unseenclasses = len(self.unseenclasses)




        self.test_feature, self.test_label = self.test_seen_feature, self.test_seen_label
        self.unknown_feature, self.unknown_label = self.test_unseen_feature, self.test_unseen_label
        self.online_unknown_feature, self.online_unknown_label = self.train_unseen_feature, self.train_unseen_label


        # label unknowns as negatives
        self.unknown_label = torch.tensor([-self.unknown_label[i] - 1 for i in range(len(self.unknown_label))])
        self.online_unknown_label = torch.tensor(
            [-self.online_unknown_label[i] - 1 for i in range(len(self.online_unknown_label))])





        self.train_known_feature = self.train_feature
        self.train_known_label = self.train_label

        self.test_known_feature = self.test_feature
        self.test_known_label = self.test_label

        self.train_unknown_feature = self.online_unknown_feature
        self.train_unknown_label = self.online_unknown_label

        self.test_unknown_feature = self.unknown_feature
        self.test_unknown_label = self.unknown_label

        #### shuffle #####

        shuffle_index = torch.randperm(len(self.train_known_feature))
        self.train_known_feature = self.train_known_feature[shuffle_index]
        self.train_known_label = self.train_known_label[shuffle_index]

        shuffle_index = torch.randperm(len(self.test_known_feature))
        self.test_known_feature = self.test_known_feature[shuffle_index]
        self.test_known_label = self.test_known_label[shuffle_index]

        shuffle_index = torch.randperm(len(self.train_unknown_feature))
        self.train_unknown_feature = self.train_unknown_feature[shuffle_index]
        self.train_unknown_label = self.train_unknown_label[shuffle_index]

        shuffle_index = torch.randperm(len(self.test_unknown_feature))
        self.test_unknown_feature = self.test_unknown_feature[shuffle_index]
        self.test_unknown_label = self.test_unknown_label[shuffle_index]


    def read_cifar10(self, opt):

        if opt.baseline_type=="softmax":
            fileware = "softmax"
            prefix = "Softmax_"

        if opt.baseline_type=="clip":
            fileware = "clip_model"
            prefix = "CLIP_"

        SAVE_PARA_PATH = "deepfeature/"+fileware+"/results/"+prefix+"cifar-10-10_split"+str(opt.split)+"_train_known_set.pth"
        self.train_known_feature, self.train_known_label = torch.load(SAVE_PARA_PATH)

        SAVE_PARA_PATH = "deepfeature/"+fileware+"/results/" + prefix+"cifar-10-10_split" + str(opt.split) + "_test_known_set.pth"
        self.test_known_feature, self.test_known_label = torch.load(SAVE_PARA_PATH)

        SAVE_PARA_PATH = "deepfeature/"+fileware+"/results/" + prefix+"cifar-10-10_split" + str(opt.split) + "_train_unknown_set.pth"
        self.train_unknown_feature, self.train_unknown_label = torch.load(SAVE_PARA_PATH)

        SAVE_PARA_PATH = "deepfeature/"+fileware+"/results/" + prefix+"cifar-10-10_split" + str(opt.split) + "_test_unknown_set.pth"
        self.test_unknown_feature, self.test_unknown_label = torch.load(SAVE_PARA_PATH)

        self.train_known_feature = self.train_known_feature.float()
        self.test_known_feature = self.test_known_feature.float()
        self.train_unknown_feature = self.train_unknown_feature.float()
        self.test_unknown_feature = self.test_unknown_feature.float()




        self.nclass = 10
        self.seenclasses = 6
        self.unseenclasses = 4

        # label unknowns as negatives
        self.train_unknown_label = torch.tensor([-self.train_unknown_label[i]-1 for i in range(len(self.train_unknown_label))])
        self.test_unknown_label = torch.tensor([-self.test_unknown_label[i] - 1 for i in range(len(self.test_unknown_label))])

    def read_matdataset_tinyimagenet(self, opt):


        if opt.baseline_type=="softmax":
            fileware = "softmax"
            prefix = "Softmax_"

        if opt.baseline_type=="clip":
            fileware = "clip_model"
            prefix = "CLIP_"

        SAVE_PARA_PATH = "deepfeature/"+fileware+"/results/"+prefix+"tinyimagenet_split"+str(opt.split)+"_train_known_set.pth"
        self.train_known_feature, self.train_known_label = torch.load(SAVE_PARA_PATH)

        SAVE_PARA_PATH = "deepfeature/"+fileware+"/results/" + prefix+"tinyimagenet_split" + str(opt.split) + "_test_known_set.pth"
        self.test_known_feature, self.test_known_label = torch.load(SAVE_PARA_PATH)

        SAVE_PARA_PATH = "deepfeature/"+fileware+"/results/" + prefix+"tinyimagenet_split" + str(opt.split) + "_train_unknown_set.pth"
        self.train_unknown_feature, self.train_unknown_label = torch.load(SAVE_PARA_PATH)

        SAVE_PARA_PATH = "deepfeature/"+fileware+"/results/" + prefix+"tinyimagenet_split" + str(opt.split) + "_test_unknown_set.pth"
        self.test_unknown_feature, self.test_unknown_label = torch.load(SAVE_PARA_PATH)

        self.train_known_feature = self.train_known_feature.float()
        self.test_known_feature = self.test_known_feature.float()
        self.train_unknown_feature = self.train_unknown_feature.float()
        self.test_unknown_feature = self.test_unknown_feature.float()

        self.nclass = 200
        self.seenclasses = 20
        self.unseenclasses = 180

        # label unknowns as negatives
        self.train_unknown_label = torch.tensor([-self.train_unknown_label[i]-1 for i in range(len(self.train_unknown_label))])
        self.test_unknown_label = torch.tensor([-self.test_unknown_label[i] - 1 for i in range(len(self.test_unknown_label))])

    # def read_cifar100(self, opt):
    #
    #
    #     SAVE_PARA_PATH = "backbone/results/"+"cifar-100_split"+str(opt.split)+"_train_known_set.pth"
    #     self.train_known_feature, self.train_known_label = torch.load(SAVE_PARA_PATH)
    #
    #     SAVE_PARA_PATH = "backbone/results/" + "cifar-100_split" + str(opt.split) + "_test_known_set.pth"
    #     self.test_known_feature, self.test_known_label = torch.load(SAVE_PARA_PATH)
    #
    #     SAVE_PARA_PATH = "backbone/results/" + "cifar-100_split" + str(opt.split) + "_train_unknown_set.pth"
    #     self.train_unknown_feature, self.train_unknown_label = torch.load(SAVE_PARA_PATH)
    #
    #     SAVE_PARA_PATH = "backbone/results/" + "cifar-100_split" + str(opt.split) + "_test_unknown_set.pth"
    #     self.test_unknown_feature, self.test_unknown_label = torch.load(SAVE_PARA_PATH)
    #
    #     SAVE_PARA_PATH = "backbone/results/" + "cifar-100_split" + str(opt.split) + "_val_known_set.pth"
    #     self.val_known_feature, self.val_known_label = torch.load(SAVE_PARA_PATH)
    #
    #     self.test_known_feature = torch.cat([self.test_known_feature, self.val_known_feature], dim=0)
    #     self.test_known_label = torch.cat([self.test_known_label, self.val_known_label], dim=0)
    #     #
    #     # self.test_known_feature = self.val_known_feature
    #     # self.test_known_label = self.val_known_label
    #
    #     self.nclass = 100
    #     self.seenclasses = 10
    #     self.unseenclasses = 90
    #
    #     # label unknowns as negatives
    #     self.train_unknown_label = torch.tensor([-self.train_unknown_label[i]-1 for i in range(len(self.train_unknown_label))])
    #     self.test_unknown_label = torch.tensor([-self.test_unknown_label[i] - 1 for i in range(len(self.test_unknown_label))])
    #
    # def read_mnist(self, opt):
    #
    #
    #     SAVE_PARA_PATH = "backbone/results/"+"mnist_split"+str(opt.split)+"_train_known_set.pth"
    #     self.train_known_feature, self.train_known_label = torch.load(SAVE_PARA_PATH)
    #
    #     SAVE_PARA_PATH = "backbone/results/" + "mnist_split" + str(opt.split) + "_test_known_set.pth"
    #     self.test_known_feature, self.test_known_label = torch.load(SAVE_PARA_PATH)
    #
    #     SAVE_PARA_PATH = "backbone/results/" + "mnist_split" + str(opt.split) + "_train_unknown_set.pth"
    #     self.train_unknown_feature, self.train_unknown_label = torch.load(SAVE_PARA_PATH)
    #
    #     SAVE_PARA_PATH = "backbone/results/" + "mnist_split" + str(opt.split) + "_test_unknown_set.pth"
    #     self.test_unknown_feature, self.test_unknown_label = torch.load(SAVE_PARA_PATH)
    #
    #     self.nclass = 10
    #     self.seenclasses = 6
    #     self.unseenclasses = 4
    #
    #     # label unknowns as negatives
    #     self.train_unknown_label = torch.tensor([-self.train_unknown_label[i]-1 for i in range(len(self.train_unknown_label))])
    #     self.test_unknown_label = torch.tensor([-self.test_unknown_label[i] - 1 for i in range(len(self.test_unknown_label))])
    #

    #
    def read_matdataset_aPaY(self, opt):

        ####################  也是极度不平衡的数据集   ########

        matcontent = sio.loadmat(opt.dataroot + "/" + "APY" + "/" + opt.image_embedding + ".mat")
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(opt.dataroot + "/" + "APY" + "/" + opt.class_embedding + "_splits.mat")

        self.all_class_num = 32



        #################################################################################################################
        if opt.split == 0:
            split_known_class = [ 7,  9, 15, 16, 20,  2, 27, 26, 14, 25]#split 0
        if opt.split == 1:
            split_known_class = [2, 11, 7, 20, 9, 17, 31, 6, 13, 24] #split 1
        if opt.split == 2:
            split_known_class = [15, 22, 24, 28, 19, 26,  0, 31,  7,  1]#split2
        if opt.split == 3:
            split_known_class = [3, 8, 20, 27, 1, 28, 6, 25, 14, 7]#split3
        if opt.split == 4:
            split_known_class = [24, 14, 4, 27, 9, 0, 3, 2, 11, 26]  # split4



        self.split_unknown = [i for i in range(1, 32 + 1) if i not in split_known_class]

        for kk in range(len(split_known_class)):
            split_known_class[kk] = split_known_class[kk] + 1


        self._att_name = []
        self._class_name = []
        self._train_ids = []
        self._test_ids = []
        self._open_ids = []
        self._open_online_ids = []
        self._image_id_label = {}
        # self.split_unknown = [i for i in range(1, self.nclass + 1) if i not in self.split_known]

        apay_split_ratio = 4/5
        random.seed(6666)

        self.class_sample_check_dict={}

        for i in range(self.all_class_num):
            self.class_sample_check_dict[i] = []

        for i in range(len(label)):
            self.class_sample_check_dict[label[i]].append(i)

        for i in range(len(self.class_sample_check_dict)):
            random.shuffle(self.class_sample_check_dict[i])
            train_len = int(len(self.class_sample_check_dict[i])*apay_split_ratio)
            if i in split_known_class:
                self._train_ids.extend(self.class_sample_check_dict[i][:train_len])
                self._test_ids.extend(self.class_sample_check_dict[i][train_len:])
            else:
                self._open_online_ids.extend(self.class_sample_check_dict[i][:train_len])
                self._open_ids.extend(self.class_sample_check_dict[i][train_len:])


        for kk in range(len(self._train_ids)):
            self._train_ids[kk] = int(self._train_ids[kk])

        self._train_ids = np.array(self._train_ids)

        for kk in range(len(self._test_ids)):
            self._test_ids[kk] = int(self._test_ids[kk])

        self._test_ids = np.array(self._test_ids)

        for kk in range(len(self._open_online_ids)):
            self._open_online_ids[kk] = int(self._open_online_ids[kk])

        self._open_online_ids = np.array(self._open_online_ids)

        for kk in range(len(self._open_ids)):
            self._open_ids[kk] = int(self._open_ids[kk])

        self._open_ids = np.array(self._open_ids)

        trainval_loc = self._train_ids
        test_seen_loc = self._test_ids
        test_unseen_loc = self._open_ids
        train_unseen_loc = self._open_online_ids


        #################################################################################################################

        scaler = preprocessing.MinMaxScaler()

        _train_feature = scaler.fit_transform(feature[trainval_loc])
        _test_seen_feature = scaler.transform(feature[test_seen_loc])
        _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
        _train_unseen_feature = scaler.transform(feature[train_unseen_loc])

        self.train_known_feature = torch.from_numpy(_train_feature).float()
        mx = self.train_known_feature.max()
        self.train_known_feature.mul_(1 / mx)

        self.seenclasses_buff = list(np.unique(torch.from_numpy(label[trainval_loc]).long().numpy()))
        self.unseenclasses_buff = list(np.unique(torch.from_numpy(label[test_unseen_loc]).long().numpy()))

        self.train_known_label = torch.tensor([self.seenclasses_buff.index(ele) for ele in list(label[trainval_loc])]).long()

        self.train_unseen_feature = torch.from_numpy(_train_unseen_feature).float()
        self.train_unseen_feature.mul_(1 / mx)
        self.train_unseen_label = torch.tensor(
            [self.unseenclasses_buff.index(ele) for ele in list(label[train_unseen_loc])]).long()

        self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
        self.test_unseen_feature.mul_(1 / mx)
        self.test_unseen_label = torch.tensor(
            [self.unseenclasses_buff.index(ele) for ele in list(label[test_unseen_loc])]).long()

        self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
        self.test_seen_feature.mul_(1 / mx)
        self.test_seen_label = torch.tensor(
            [self.seenclasses_buff.index(ele) for ele in list(label[test_seen_loc])]).long()

        self.seenclasses = torch.from_numpy(np.unique(torch.from_numpy(label[trainval_loc]).long().numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(torch.from_numpy(label[test_unseen_loc]).long().numpy()))

        self.seenclasses = len(self.seenclasses)
        self.unseenclasses = len(self.unseenclasses)

        self.test_known_feature, self.test_known_label = self.test_seen_feature, self.test_seen_label
        self.test_unknown_feature, self.test_unknown_label = self.test_unseen_feature, self.test_unseen_label
        self.train_unknown_feature, self.train_unknown_label = self.train_unseen_feature, self.train_unseen_label

    def read_matdataset_SUN(self, opt):

        matcontent = sio.loadmat(opt.dataroot + "/" + "SUN" + "/" + opt.image_embedding + ".mat")
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(opt.dataroot + "/" + "SUN" + "/" + opt.class_embedding + "_splits.mat")

        self.all_class_num = 717
        self.all_att_num = 102


        #################################################################################################################
        if opt.split == 0:
            split_known_class = list(range(50))#split 0
        if opt.split == 1:
            split_known_class = list(range(300,350)) #split 1
        if opt.split == 2:
            split_known_class = list(range(600,650))#split2
        if opt.split == 3:
            split_known_class = list(range(0,100))#split2



        self.split_unknown = [i for i in range(1, 717 + 1) if i not in split_known_class]

        index = torch.randperm(self.all_att_num)
        chosen_att = index[:self.all_att_num]


        for kk in range(len(split_known_class)):
            split_known_class[kk] = split_known_class[kk] + 1

        self._att_name = []
        self._class_name = []
        self._train_ids = []
        self._test_ids = []
        self._open_ids = []
        self._open_online_ids = []
        self._image_id_label = {}
        # self.split_unknown = [i for i in range(1, self.nclass + 1) if i not in self.split_known]

        SUN_split_ratio = 4/5
        random.seed(6666)

        self.class_sample_check_dict={}

        for i in range(self.all_class_num):
            self.class_sample_check_dict[i] = []

        for i in range(len(label)):
            self.class_sample_check_dict[label[i]].append(i)

        for i in range(len(self.class_sample_check_dict)):
            random.shuffle(self.class_sample_check_dict[i])
            train_len = int(len(self.class_sample_check_dict[i]) * SUN_split_ratio)
            if i in split_known_class:
                self._train_ids.extend(self.class_sample_check_dict[i][:train_len])
                self._test_ids.extend(self.class_sample_check_dict[i][train_len:])
            else:
                self._open_online_ids.extend(self.class_sample_check_dict[i][:train_len])
                self._open_ids.extend(self.class_sample_check_dict[i][train_len:])




        for kk in range(len(self._train_ids)):
            self._train_ids[kk] = int(self._train_ids[kk])

        self._train_ids = np.array(self._train_ids)

        for kk in range(len(self._test_ids)):
            self._test_ids[kk] = int(self._test_ids[kk])

        self._test_ids = np.array(self._test_ids)

        for kk in range(len(self._open_online_ids)):
            self._open_online_ids[kk] = int(self._open_online_ids[kk])

        self._open_online_ids = np.array(self._open_online_ids)

        for kk in range(len(self._open_ids)):
            self._open_ids[kk] = int(self._open_ids[kk])

        self._open_ids = np.array(self._open_ids)

        trainval_loc = self._train_ids
        test_seen_loc = self._test_ids
        test_unseen_loc = self._open_ids
        train_unseen_loc = self._open_online_ids



        #################################################################################################################
        scaler = preprocessing.MinMaxScaler()

        _train_feature = scaler.fit_transform(feature[trainval_loc])
        _test_seen_feature = scaler.transform(feature[test_seen_loc])
        _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
        _train_unseen_feature = scaler.transform(feature[train_unseen_loc])

        self.train_known_feature = torch.from_numpy(_train_feature).float()
        mx = self.train_known_feature.max()
        self.train_known_feature.mul_(1 / mx)

        self.seenclasses_buff = list(np.unique(torch.from_numpy(label[trainval_loc]).long().numpy()))
        self.unseenclasses_buff = list(np.unique(torch.from_numpy(label[test_unseen_loc]).long().numpy()))

        self.train_known_label = torch.tensor(
            [self.seenclasses_buff.index(ele) for ele in list(label[trainval_loc])]).long()

        self.train_unseen_feature = torch.from_numpy(_train_unseen_feature).float()
        self.train_unseen_feature.mul_(1 / mx)
        self.train_unseen_label = torch.tensor(
            [self.unseenclasses_buff.index(ele) for ele in list(label[train_unseen_loc])]).long()

        self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
        self.test_unseen_feature.mul_(1 / mx)
        self.test_unseen_label = torch.tensor(
            [self.unseenclasses_buff.index(ele) for ele in list(label[test_unseen_loc])]).long()

        self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
        self.test_seen_feature.mul_(1 / mx)
        self.test_seen_label = torch.tensor(
            [self.seenclasses_buff.index(ele) for ele in list(label[test_seen_loc])]).long()

        self.seenclasses = torch.from_numpy(np.unique(torch.from_numpy(label[trainval_loc]).long().numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(torch.from_numpy(label[test_unseen_loc]).long().numpy()))

        self.seenclasses = len(self.seenclasses)
        self.unseenclasses = len(self.unseenclasses)

        self.test_known_feature, self.test_known_label = self.test_seen_feature, self.test_seen_label
        self.test_unknown_feature, self.test_unknown_label = self.test_unseen_feature, self.test_unseen_label
        self.train_unknown_feature, self.train_unknown_label = self.train_unseen_feature, self.train_unseen_label





    def _open_split(self):

        flag = 1
        flag2 = 0

        while(flag):
            for k in range(len(self._train_ids)):
                if int(self._image_id_label[self._train_ids[k]]) not in self.split_known:

                    if int(self._image_id_label[self._train_ids[k]]) in self.split_unknown:
                        self._open_online_ids.append(self._train_ids[k])
                    self._train_ids.remove(self._train_ids[k])
                    flag2 = 1
                    break
            if flag2 == 0:
                flag = 0
            flag2 = 0



        flag = 1
        flag2 = 0

        while (flag):
            for k in range(len(self._test_ids)):
                if int(self._image_id_label[self._test_ids[k]]) not in self.split_known:

                    if int(self._image_id_label[self._test_ids[k]]) in self.split_unknown:
                        self._open_ids.append(self._test_ids[k])
                    self._test_ids.remove(self._test_ids[k])
                    flag2 = 1
                    break
            if flag2 == 0:
                flag = 0
            flag2 = 0



    def _train_test_split(self):

        for line in open(self.train_test_split_file):
            image_id, label = line.strip('\n').split()
            if label == '1':
                self._train_ids.append(image_id)
            elif label == '0':
                self._test_ids.append(image_id)
            else:
                raise Exception('label error')

    def _get_id_to_label(self):
        for line in open(self.image_class_labels_file):
            image_id, class_id = line.strip('\n').split()
            self._image_id_label[image_id] = class_id