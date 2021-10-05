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
from test import test

import glob
import shutil as sh
from haven import haven_chk as hc
import torch.distributed as dist
# import wandb
import numpy as np
import hashlib
from PIL import Image
torch.backends.cudnn.benchmark = True


def trainval(exp_dict, savedir, args):
    """Main."""
    # ==========================
    # load datasets
    # ==========================
    train_set = datasets.get_dataset('train', args.datadir, exp_dict)
    train_set_pred = datasets.get_dataset('train_pred', args.datadir, exp_dict)

    val_set = datasets.get_dataset('val', args.datadir, exp_dict)
    val_set_imagenet = datasets.get_dataset('val', args.datadir, exp_dict, "imagenet")
    

    # test_set = datasets.get_dataset(exp_dict["dataset_test"], 'test', exp_dict)
    if exp_dict.get("rescale_lr", False):
        exp_dict['lr'] = exp_dict['lr'] * exp_dict["batch_size"] / 256.

    # ==========================
    # get dataloaders
    # ==========================
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=exp_dict["batch_size"],
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False)
    train_loader_pred = torch.utils.data.DataLoader(
        train_set_pred,
        batch_size=exp_dict["batch_size"],
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False)
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=exp_dict["batch_size"],
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True)
    val_loader_imagenet = torch.utils.data.DataLoader(
        val_set_imagenet,
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
        score_list = hu.load_pkl(score_list_path)
        s_epoch = score_list[-1]['epoch'] + 1
    else:
        # restart experiment
        score_list = []
        s_epoch = 0

    # Run training and validation
    for epoch in range(s_epoch, exp_dict["max_epoch"]):
        score_dict = {"epoch": epoch}
        #score_dict.update(model.get_lr())

        # train
        if exp_dict["model"] == 'iterative_real':
            score_dict.update(model.fit(epoch, train_loader, val_loader))
            # Add score_dict to score_list
            score_list += [score_dict]

        elif exp_dict["model"] == 'iterative_real_k1k2':
            score_dict_list = model.fit(epoch, train_loader, val_loader, train_loader_pred)
            score_list += score_dict_list
        else:
            score_dict.update(model.train_on_loader(epoch, train_loader))
            #validate
            score_dict.update(model.val_on_loader_wv(epoch, val_loader))
            score_dict.update(model.val_on_loader(epoch, val_loader_imagenet))

            # Add score_dict to score_list
            score_list += [score_dict]

        # Report
        score_df = pd.DataFrame(score_list)
        print(score_df.tail())
        # Save checkpoint
        hu.save_pkl(savedir + "/score_list.pkl", score_list)
        hu.torch_save(savedir + "/checkpoint.pth", model.get_state_dict())
        print("Saved: %s" % savedir)

        # Save best checkpoint
        if exp_dict.get('early_stopping', False):
            if score_dict['val_loss'] <= score_df['val_loss'][:-1].min():
                hu.save_pkl(savedir + "/score_list_best.pkl", score_list)
                hu.torch_save(savedir + "/checkpoint_best.pth",
                              model.get_state_dict())
                print("Saved Best: %s" % savedir)
            # Check for end of training conditions
            elif exp_dict.get('early_stopping') < (epoch - score_df['val_loss'].argmin()):
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--exp_group_list', nargs="+", default="resnet", help='name of an experiment in exp_configs.py')
    parser.add_argument('-sb', '--savedir_base', default="/mnt/home/haven_output", help='folder where logs will be saved')
    parser.add_argument('-nw', '--num_workers', type=int, default=4)
    parser.add_argument('-d', '--datadir', type=str, default="./data")
    parser.add_argument('-rr', '--relabelroot', type=str, default="/mnt/public/datasets/imagenet/relabel/imagenet_efficientnet_l2_sz475_top5")
    parser.add_argument("-r", "--reset",  default=1, type=int, help='Overwrite previous results')
    parser.add_argument("-ei", "--exp_id", default=None)
    parser.add_argument("-j", "--job_scheduler", type=str,  default=None, help="If 1, runs in toolkit in parallel")
    parser.add_argument("-v",  default="results.ipynb", help="orkestrator")
    parser.add_argument("--python_binary", default='/mnt/home/jaxvenv/bin/python', help='path to your python executable')
    parser.add_argument("--test", '-t', type=int, default=0)

    args, unknown = parser.parse_known_args()

    import exp_configs
    import job_configs
    try:
        job_configs.JOB_CONFIG['resources']['gpu'] = exp_configs.EXP_GROUPS[args.exp_group_list[0]][0]['ngpu']
    except:
        pass
    if args.test == 1:
        hw.run_wizard(func=test,
                  exp_groups=exp_configs.EXP_GROUPS,
                  job_config=job_configs.JOB_CONFIG,
                  python_binary_path=args.python_binary,
                  use_threads=True,
                  args=args)
    else:
        hw.run_wizard(func=trainval,
                    exp_groups=exp_configs.EXP_GROUPS,
                    job_config=job_configs.JOB_CONFIG,
                    python_binary_path=args.python_binary,
                    use_threads=True,
                    args=args)