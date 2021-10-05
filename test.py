import torch
import argparse
import pandas as pd
import sys
import os
import torch
from torch import nn
from torch.nn import functional as F
import tqdm
import pprint
from models import get_model
import torchvision
from haven import haven_utils as hu
from haven import haven_wizard as hw
import  datasets
from torch.utils.data import DataLoader
import exp_configs
from torch.utils.data.sampler import RandomSampler

import glob
import shutil as sh
from haven import haven_chk as hc
import torch.distributed as dist
# import wandb
import numpy as np
import hashlib
from PIL import Image
from copy import deepcopy
torch.backends.cudnn.benchmark = True


def test(exp_dict, savedir, args):
    """Main."""
    # ==========================
    # load datasets
    # ==========================
    # train_set = datasets.get_dataset('train', args.datadir, exp_dict)
    # train_set_pred = datasets.get_dataset('train_pred', args.datadir, exp_dict)

    val_set = datasets.get_dataset('val', args.datadir, exp_dict)

    # test_set = datasets.get_dataset(exp_dict["dataset_test"], 'test', exp_dict)
    if exp_dict.get("rescale_lr", False):
        exp_dict['lr'] = exp_dict['lr'] * exp_dict["batch_size"] / 256.

    # ==========================
    # get dataloaders
    # ==========================
    # train_loader = torch.utils.data.DataLoader(
    #     train_set,
    #     batch_size=exp_dict["batch_size"],
    #     shuffle=True,
    #     num_workers=args.num_workers,
    #     drop_last=False)
    # train_loader_student = torch.utils.data.DataLoader(
    #     train_set,
    #     batch_size=exp_dict["batch_size"],
    #     shuffle=True,
    #     num_workers=args.num_workers,
    #     drop_last=False)
    # train_loader_pred = torch.utils.data.DataLoader(
    #     train_set_pred,
    #     batch_size=exp_dict["batch_size"],
    #     shuffle=True,
    #     num_workers=args.num_workers,
    #     drop_last=False)
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=exp_dict["batch_size"],
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True)
    # test_loader = torch.utils.data.DataLoader(
    #     test_set,
    #     batch_size=exp_dict["batch_size"],
    #     shuffle=False,
    #     num_workers=exp_dict["num_workers"],
    #     drop_last=False)

    # ==========================
    # create model and trainer
    # ==========================

    # writer = tensorboardX.SummaryWriter(savedir) \
    #       if tensorboard == 1 else None

    model = get_model(exp_dict)
                                 

    # model.opt = optimizers.get_optim(exp_dict['opt'], model)
    model_path = os.path.join(savedir, "checkpoint.pth")
    score_list_path = os.path.join(savedir, "score_list.pkl")

    if os.path.exists(score_list_path):
        # resume experiment
        model.set_state_dict(hu.torch_load(model_path))
        score_list = []
        #score_list = hu.load_pkl(score_list_path)
        #s_epoch = score_list[-1]['iter'] + 1
        s_epoch = 0
    else:
        # restart experiment
        score_list = []
        s_epoch = 0

    # create dataloader-iterator
    # data_iter_student = iter(train_loader_student) 
    #data_iter = iter(train_loader) 
    # iterate over dataset
    # alternatively you could use while(True) 
    
           

    ret_dict = {}
    print("Testing teacher and student")
    ret_dict.update(model.test_on_loader(0, val_loader))
    #ret_dict = {k.replace("val_", "val_studentmain_"): v for k, v in ret_dict.items()}
    model.top1_acc = ret_dict["val_acc_1"]

    score_list += [ret_dict]

    score_df = pd.DataFrame(score_list)
    print(score_df.tail())
    # Report            
    # Save checkpoint
    hu.save_pkl(savedir + "/score_list_test.pkl", score_list)
    print("Saved: %s" % savedir)