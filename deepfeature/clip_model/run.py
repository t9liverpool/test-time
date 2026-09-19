import os
import argparse
import datetime
import time
import pandas as pd
import importlib

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.backends.cudnn as cudnn


from data_prepare import get_class_splits, get_datasets
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np

import evaluation

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, roc_auc_score

import clip
from torchvision.datasets import CIFAR100



def get_default_hyperparameters(args):

    """
    Adjusts args to match parameters used in paper: https://arxiv.org/abs/2110.06207
    """

    args.device = torch.device("cuda:0")

    args.dataset = "cifar-10-10"
    args.dataset = "cifar-10-100"
    args.dataset = "tinyimagenet"
    # args.dataset = "imagenet1k"
    # args.dataset = "cifar10-svhn"
    args.split_idx = 3

    args.optim=None


    hyperparameter_path = './paper_hyperparameters.csv'
    df = pd.read_csv(hyperparameter_path)

    if args.dataset=="cifar10-svhn":
        hyperparams = df.loc[df['Dataset'] == "cifar-10-10"].values[0][2:]

    else:
        hyperparams = df.loc[df['Dataset'] == args.dataset].values[0][2:]




    # -----------------
    # DATASET / LOSS specific hyperparams
    # -----------------
    args.image_size, _, _, _, _, args.batch_size = hyperparams



    if args.dataset in ["cifar-10-10", "cifar-10-100", "cifar10-svhn"]:

        args.seed = 0
        args.transform = "pytorch-cifar"

    if args.dataset in ["tinyimagenet", "imagenet1k"]:

        args.seed = 0
        args.transform = "pytorch-tinyimagenet"

    if args.dataset in ["imagenet1k"]:

        args.seed = 0
        args.transform = "pytorch-imagenet1k"



    return args


def get_features(options, args):

    # -----------------------------
    # DATALOADERS
    # -----------------------------

    train_known_loader = dataloaders['train']
    train_unknown_loader = dataloaders['train_unknown']
    test_known_loader = dataloaders['test_known']
    test_unknown_loader = dataloaders['test_unknown']



    # -----------------------------
    # MODEL
    # -----------------------------
    print("Creating model: {}".format(options['model']))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, preprocess = clip.load("ViT-B/32", device=device)

    net.eval()
    net.to(args.device)

    # if args.dataset in ["cifar-10-10", "cifar-10-100"]:
    #
    #     label_prompt = ["a photo of a airplane", "a photo of a automobile", "a photo of a bird", "a photo of a cat", "a photo of a deer",
    #                     "a photo of a dog", "a photo of a frog", "a photo of a horse", "a photo of a ship", "a photo of a truck"]
    #
    #     label_prompt = [label_prompt[i] for i in args.train_classes]
    #
    #     text = clip.tokenize(label_prompt).to(device)
    #
    #     text_features = net.encode_text(text)
    #
    #
    #     # 非tiny可以简单测一下clip的性能
    #     with torch.no_grad():
    #
    #         # -----------------------------
    #         # TEST
    #         # -----------------------------
    #
    #         known_scores_av = []
    #         known_scores_softmax = []
    #
    #         accuracy = AverageMeter()
    #
    #         for batch_idx, data in enumerate(test_known_loader):
    #
    #             inputv, labelv, _ = data
    #             inputv = inputv.cuda()
    #             labelv = labelv.cuda()
    #
    #
    #             logits_per_image, logits_per_text = net(inputv, text)
    #             probs = logits_per_image.softmax(dim=-1)
    #
    #             _, preds = torch.max(probs, 1)
    #
    #             known_scores_av.extend(
    #                 [torch.max(logits_per_image.cpu(), dim=1)[0][i].cpu().numpy() for i in range(len(inputv))])
    #
    #             known_scores_softmax.extend(
    #                 [torch.max(probs.cpu(), dim=1)[0][i].cpu().numpy() for i in range(len(inputv))])
    #
    #             accuracy.update(torch.sum(preds == labelv).item(), len(inputv))
    #
    #         unknown_scores_av = []
    #         unknown_scores_softmax = []
    #
    #         for batch_idx, data in enumerate(test_unknown_loader):
    #
    #             inputv, labelv, _ = data
    #             inputv = inputv.cuda()
    #             labelv = labelv.cuda()
    #
    #             logits_per_image, logits_per_text = net(inputv, text)
    #             probs = logits_per_image.softmax(dim=-1)
    #
    #             _, preds = torch.max(probs, 1)
    #
    #             unknown_scores_av.extend(
    #                 [torch.max(logits_per_image.cpu(), dim=1)[0][i].cpu().numpy() for i in range(len(inputv))])
    #
    #             unknown_scores_softmax.extend(
    #                 [torch.max(probs.cpu(), dim=1)[0][i].cpu().numpy() for i in range(len(inputv))])
    #
    #         auc_score_av = compute_roc(known_scores_av, unknown_scores_av)
    #         auc_score_softmax = compute_roc(known_scores_softmax, unknown_scores_softmax)
    #         print("ACC:", accuracy.avg, "AV_ROC:", auc_score_av, "SOFTMAX_ROC", auc_score_softmax)

    # -----------------------------
    # GET FEATURES
    # -----------------------------
    with torch.no_grad():
        get_features = None
        get_labels = None

        for batch_idx, data in enumerate(train_known_loader):


            if args.dataset in ["imagenet1k"]:
                inputv, labelv = data
            else:
                inputv, labelv, _ = data

            inputv = inputv.cuda()
            labelv = labelv.cuda()


            features = net.encode_image(inputv).cpu()

            if get_features == None:
                get_features = features.cpu()
                get_labels = labelv.cpu()
            else:
                get_features = torch.cat([get_features, features.cpu()], dim=0)
                get_labels = torch.cat([get_labels, labelv.cpu()], dim=0)

        print("训练样本数目：", len(get_labels))
        SAVE_PARA_PATH = "./results/" + args.loss+"_"+args.dataset + "_split" + str(args.split_idx) + "_train_known_set.pth"
        torch.save([get_features, get_labels], SAVE_PARA_PATH)


        get_features = None
        get_labels = None

        for batch_idx, data in enumerate(test_known_loader):
            if args.dataset in ["imagenet1k"]:
                inputv, labelv = data
            else:
                inputv, labelv, _ = data
            inputv = inputv.cuda()
            labelv = labelv.cuda()

            features = net.encode_image(inputv).cpu()

            if get_features == None:
                get_features = features.cpu()
                get_labels = labelv.cpu()
            else:
                get_features = torch.cat([get_features, features.cpu()], dim=0)
                get_labels = torch.cat([get_labels, labelv.cpu()], dim=0)

        print("测试样本数目：", len(get_labels))
        SAVE_PARA_PATH = "./results/" + args.loss+"_"+args.dataset + "_split" + str(args.split_idx) + "_test_known_set.pth"
        torch.save([get_features, get_labels], SAVE_PARA_PATH)

        get_features = None
        get_labels = None

        for batch_idx, data in enumerate(train_unknown_loader):
            if args.dataset in ["imagenet1k"]:
                inputv, labelv = data
            else:
                inputv, labelv, _ = data
            inputv = inputv.cuda()
            labelv = labelv.cuda()


            features = net.encode_image(inputv).cpu()

            if get_features == None:
                get_features = features.cpu()
                get_labels = labelv.cpu()
            else:
                get_features = torch.cat([get_features, features.cpu()], dim=0)
                get_labels = torch.cat([get_labels, labelv.cpu()], dim=0)

        print("OOD样本数目：", len(get_labels))
        SAVE_PARA_PATH = "./results/" + args.loss+"_"+args.dataset + "_split" + str(args.split_idx) + "_train_unknown_set.pth"
        torch.save([get_features, get_labels], SAVE_PARA_PATH)

        get_features = None
        get_labels = None

        for batch_idx, data in enumerate(test_unknown_loader):
            if args.dataset in ["imagenet1k"]:
                inputv, labelv = data
            else:
                inputv, labelv, _ = data
            inputv = inputv.cuda()
            labelv = labelv.cuda()

            features = net.encode_image(inputv).cpu()

            if get_features == None:
                get_features = features.cpu()
                get_labels = labelv.cpu()
            else:
                get_features = torch.cat([get_features, features.cpu()], dim=0)
                get_labels = torch.cat([get_labels, labelv.cpu()], dim=0)

        print("在线已知类样本数目：", len(get_labels))
        SAVE_PARA_PATH = "./results/" + args.loss+"_"+args.dataset + "_split" + str(args.split_idx) + "_test_unknown_set.pth"
        torch.save([get_features, get_labels], SAVE_PARA_PATH)

        print("finished")
        exit()




class AverageMeter(object):
    """Computes and stores the average and current value.

       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

def compute_roc(known_scores, unknown_scores):
    y_true = np.array([1] * len(known_scores) + [0] * len(unknown_scores))
    y_score = np.concatenate([known_scores, unknown_scores])
    # fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)
    return auc_score


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    exp_root = "./checkpoints"


    print('NOTE: Using default hyper-parameters...')
    args = get_default_hyperparameters(args)

    args.model = "CLIP"
    args.loss = "CLIP"


    img_size = args.image_size
    results = dict()


    for i in range(1):

        # ------------------------
        # INIT
        # ------------------------

        args.train_classes, args.open_set_classes = get_class_splits(args.dataset, args.split_idx)

        img_size = args.image_size

        args.save_name = '{}_{}'.format(args.model, args.dataset)

        # ------------------------
        # DATASETS
        # ------------------------


        datasets = get_datasets(args.dataset, transform=args.transform, train_classes=args.train_classes,
                                open_set_classes=args.open_set_classes, balance_open_set_eval=True,
                                image_size=args.image_size, seed=args.seed,
                                args=args)



        # ------------------------
        # DATALOADER
        # ------------------------
        dataloaders = {}
        for k, v, in datasets.items():
            shuffle = True if k == 'train' else False
            dataloaders[k] = DataLoader(v, batch_size=args.batch_size,
                                        shuffle=shuffle, sampler=None, num_workers=4)



        # ------------------------
        # SAVE PARAMS
        # ------------------------
        options = vars(args)
        options.update(
            {
                'item':     i,
                'known':    args.train_classes,
                'unknown':  args.open_set_classes,
                'img_size': img_size,
                'dataloaders': dataloaders,
                'num_classes': len(args.train_classes)
            }
        )

        # ------------------------
        # TRAIN
        # ------------------------

        get_features(options, args)