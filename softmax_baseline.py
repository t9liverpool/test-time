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
from sklearn.cluster import KMeans
import copy
import time
import sklearn
from torch.nn import DataParallel
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, roc_auc_score
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from progress.bar import Bar as Bar
import numpy as np
from util import FeatDataset, AverageMeter, compute_roc


def train(baseline_type, input_dim, nclass, opt, train_known_feature, train_known_label,test_known_feature,
             test_known_label, test_unknown_feature, test_unknown_label):

    trainset = FeatDataset(data=[train_known_feature, train_known_label])
    testset = FeatDataset(data=[test_known_feature, test_known_label])
    unknownset = FeatDataset(data=[test_unknown_feature, test_unknown_label])

    trainloader= DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)
    unknownloader = DataLoader(unknownset, batch_size=128, shuffle=False, num_workers=0)

    plain_model = LINEAR_LOGSOFTMAX(input_dim, nclass).to(opt.device)
    plain_model_optimizer = torch.optim.SGD(plain_model.parameters(), lr=0.01, momentum=0.9,
                                      weight_decay=5e-4,
                                      nesterov=True)




    if baseline_type == "softmax":
        prefix = "plain_softmax_"
    if baseline_type == "clip":
        prefix = "clip_softmax_"

    if os.path.exists('./checkpoints/' + prefix + opt.dataset + "_" + str(opt.split) + '.pth'):
        # if False:
        print('读取./checkpoints/' + prefix + opt.dataset + "_" + str(opt.split) + '.pth')
        this_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        plain_model.load_state_dict(
            torch.load('./checkpoints/' + prefix + opt.dataset + "_" + str(opt.split) + '.pth', map_location=this_device))
    else:

        plain_model.to(opt.device)


        training_epoch = 50

        for epoch in range(training_epoch):

            print("需要调整epoch:", epoch, training_epoch)
            # for epoch in range(1):

            train_correct = AverageMeter()
            for batch_idx, samples in enumerate(trainloader):
                inputv, labelv = samples
                plain_model.zero_grad()

                inputv = inputv.float()
                labelv = labelv.long()

                output = plain_model(inputv.to(opt.device), features=True)
                loss = nn.CrossEntropyLoss()(output.to(opt.device), labelv.to(opt.device))

                loss.backward()
                plain_model_optimizer.step()

                with torch.no_grad():
                    _, preds = torch.max(output, 1)
                    train_correct.update(preds.eq(labelv.to(opt.device)).sum().item(), len(inputv.to(opt.device)))

            if opt.debug_mode:
                print("train_acc:", train_correct.avg)

            if opt.debug_mode:
                with torch.no_grad():
                    val(False, plain_model, testloader, unknownloader, opt
                        )
        # with torch.no_grad():
        #     val(False, plain_model, testloader, unknownloader, test_unknown_feature_classwise,
        #         )
        SAVE_PARA_PATH = './checkpoints/' + prefix + opt.dataset + "_" + str(opt.split) + '.pth'
        torch.save(plain_model.state_dict(), SAVE_PARA_PATH)

    return plain_model





def val(final_flag, plain_model, testloader, unknownloader, opt):

    known_scores_av = []
    known_scores_softmax = []
    known_labels = []
    known_preds = []
    unknown_preds = []
    test_correct = AverageMeter()
    test_preds = []
    open_preds = []

    for batch_idx, samples in enumerate(testloader):

        inputv, labelv = samples

        inputv = inputv.float()

        output = torch.nn.functional.softmax(plain_model(inputv.to(opt.device), features=True), dim=1)

        av_output = plain_model(inputv.to(opt.device), features=True)

        known_scores_av.extend(
            [torch.max(av_output.cpu(), dim=1)[0][i].cpu().numpy() for i in range(len(inputv))])

        known_scores_softmax.extend(
            [torch.max(output.cpu(), dim=1)[0][i].cpu().numpy() for i in range(len(inputv))])

        known_labels.extend([labelv[i].cpu().numpy() for i in range(len(inputv))])

        _, preds = torch.max(av_output, 1)

        known_preds.extend([preds[i].cpu().numpy() for i in range(len(inputv))])


        test_preds.extend([preds[i].cpu().numpy() for i in range(len(inputv))])

        test_correct.update(preds.eq(labelv.to(opt.device)).sum().item(), len(inputv.to(opt.device)))

        # print(labelv,preds)

    unknown_scores_av = []
    unknown_scores_softmax = []

    for batch_idx, samples in enumerate(unknownloader):

        inputv, labelv = samples

        inputv = inputv.float()

        output = torch.nn.functional.softmax(plain_model(inputv.to(opt.device), features=True), dim=1)

        av_output = plain_model(inputv.to(opt.device), features=True)

        unknown_scores_av.extend(
            [torch.max(av_output.cpu(), dim=1)[0][i].cpu().numpy() for i in range(len(inputv))])

        unknown_scores_softmax.extend(
            [torch.max(output.cpu(), dim=1)[0][i].cpu().numpy() for i in range(len(inputv))])

        _, preds = torch.max(av_output, 1)

        unknown_preds.extend([preds[i].cpu().numpy() for i in range(len(inputv))])


        open_preds.extend([preds[i].cpu().numpy() for i in range(len(inputv))])

    auc_score_av = compute_roc(known_scores_av, unknown_scores_av)
    auc_score_softmax = compute_roc(known_scores_softmax, unknown_scores_softmax)


    print("acc:", test_correct.avg, "softmax_auroc:", auc_score_softmax, "av_acroc:", auc_score_av)



    # if final_flag:
    if True:

        print("再来logit")

        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~遍历距离~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        known_scores_av = torch.tensor(
            [float(known_scores_av[ii]) for ii in range(len(known_scores_av))])
        unknown_scores_av = torch.tensor(
            [float(unknown_scores_av[ii]) for ii in range(len(unknown_scores_av))])


        TPR = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
        TNR = []

        sorted_score, _ = torch.sort(known_scores_av, descending=True)

        thresholds_logits = [sorted_score[int(len(sorted_score) * x)] for x in TPR]

        logit_95 = None

        for jj, threshold in enumerate(thresholds_logits):
            TP = sum((known_scores_av > threshold))
            TN = sum((unknown_scores_av < threshold))

            p_account = len(known_scores_av)
            n_account = len(unknown_scores_av)

            p_acc = TP / p_account
            n_acc = TN / n_account


            if logit_95 == None:
                logit_95 = n_acc

            print(
                "softmax_baseline:", "已知类样本数量", len(known_scores_av), "未知类样本数量", len(unknown_scores_av),
                "P_ACC:", p_acc,
                "N_ACC:", n_acc)

        print("logit结束啦")


        print("首先来softmax")

        # print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~遍历距离~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        known_scores_softmax = torch.tensor(
            [float(known_scores_softmax[ii]) for ii in range(len(known_scores_softmax))])
        unknown_scores_softmax = torch.tensor(
            [float(unknown_scores_softmax[ii]) for ii in range(len(unknown_scores_softmax))])

        TPR = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
        TNR = []

        sorted_score, _ = torch.sort(known_scores_softmax, descending=True)

        thresholds_softmax = [sorted_score[int(len(sorted_score) * x)] for x in TPR]


        softmax_95 = None

        for jj, threshold in enumerate(thresholds_softmax):

            TP = sum((known_scores_softmax > threshold))
            TN = sum((unknown_scores_softmax < threshold))

            p_account = len(known_scores_softmax)
            n_account = len(unknown_scores_softmax)

            p_acc = TP/p_account
            n_acc = TN/n_account

            if softmax_95 == None:
                softmax_95 = n_acc


            print(
            "softmax_baseline:", "已知类样本数量", len(known_scores_softmax), "未知类样本数量", len(unknown_scores_softmax),
              "P_ACC:", p_acc,
            "N_ACC:", n_acc)

        print("softmax结束啦")





    if softmax_95 > logit_95:

        print("选择softmax阈值")
        return thresholds_softmax
    else:
        print("选择logit阈值")
        return thresholds_logits


class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)

    def forward(self, x, features=False):
        if features:
            return self.fc(x)

        else:
            print("错误输出，不要输出logsoftmax，而是直接softmax！！")


