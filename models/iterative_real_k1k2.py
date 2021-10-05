import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones import get_backbone
from .optimizers import get_optimizer
import torch.optim 
from tqdm import tqdm
from .supervised_real import Model as SupervisedReal
from copy import deepcopy

class SoftTargetCrossEntropy(nn.Module):
    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()
        
class Model(SupervisedReal):
    """Trains a backbone using an episodic scheme on multiple GPUs"""

    def __init__(self, exp_dict):
        """ Constructor
        Args:
            backbone: architecture to train
            nclasses: number of output classes
            exp_dict: reference to dictionary with the hyperparameters
        """
        super().__init__(exp_dict)
        self.epoch = 0
        self.teacher = get_backbone(exp_dict).cuda()
        self.teacher.load_state_dict(deepcopy(self.backbone.module.state_dict()))
        self.teacher = torch.nn.DataParallel(self.teacher, list(range(self.exp_dict["ngpu"]))) 
        
    def train_on_batch(self, batch):
        """Computes the loss of a batch
        
        Args:
            batch (tuple): Inputs and labels
        
        Returns:
            loss: Loss on the batch
        """
        if self.exp_dict.get('reweight', '') == 'uniform':
            idx, images, labels, weights = batch
            weights = weights * weights.shape[0] / weights.sum()
        else:
            idx, images, labels = batch
            weights = torch.ones(labels.shape[0])
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            weights = weights.cuda(non_blocking=True)

        # compute output
        output = self.backbone(images)
        if self.sigmoid_way:
            labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=self.exp_dict["num_classes"])
            if self.exp_dict["label_smoothing"] > 0:
                labels_one_hot = labels_one_hot * self.exp_dict["label_smoothing"] + (1 - labels_one_hot) * (1 - self.exp_dict["label_smoothing"])
            if self.rescale_sigmoid:
                # tw = labels_one_hot*(self.exp_dict['num_classes'] - 1) + 1.0
                # loss = (self.criterion(output, labels_one_hot.type_as(output))*tw).mean()
                loss = self.criterion(output, labels_one_hot.type_as(output), weight=weights).sum(-1).mean()
            else:
                loss = (self.criterion(output, labels_one_hot.type_as(output), weight=weights)).mean()
            entropy = (output > 0).sum(-1).float().mean()
        else:
            loss = F.cross_entropy(output, labels)
            #loss = self.criterion(output, labels, weight=weights)
            entropy = (F.softmax(output) > 0.5).sum(-1).float().mean()

         # (F.sigmoid(output) * torch.log(F.sigmoid(output)+1e-6)).sum(-1).mean()
        return loss , entropy

    def train_on_preds(self, batch):
        """Computes the loss of a batch
        
        Args:
            batch (tuple): Inputs and labels
        
        Returns:
            loss: Loss on the batch
        """
        if self.exp_dict.get('reweight', '') == 'uniform':
            idx, images, labels, weights = batch
        else:
            idx, images, labels = batch
            weights = torch.ones(labels.shape[0])
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            weights = weights.cuda(non_blocking=True)

        # compute output
        self.backbone.train()
        output = self.backbone(images)
        with torch.no_grad():
            self.teacher.eval()
            targets_iter = self.teacher(images)
            self.teacher.train()
            #targets_iter = torch.sigmoid(targets_iter)
        if self.exp_dict["sigmoid_way"]:
            if self.exp_dict["sample_teacher"]:
                distribution = torch.distributions.bernoulli.Bernoulli(logits=targets_iter/(self.exp_dict["temperature"]+1e-5))
                targets_iter = distribution.sample().bool()
            else:
                targets_iter = torch.sigmoid(targets_iter) > self.exp_dict["binary_threshold"]
            
        
            labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=self.exp_dict["num_classes"])
            if self.exp_dict["iterative_groundtruth"] == True:
                new_labels = (targets_iter | labels_one_hot.bool()).type_as(output)
            else:
                new_labels = (targets_iter).type_as(output)

            if self.exp_dict["label_smoothing"] > 0:
                new_labels = new_labels * self.exp_dict["label_smoothing"] + (1 - new_labels) * (1 - self.exp_dict["label_smoothing"])
            if self.rescale_sigmoid:
                    # tw = labels_one_hot*(self.exp_dict['num_classes'] - 1) + 1.0
                    # loss = (self.criterion(output, labels_one_hot.type_as(output))*tw).mean()
                    loss = self.criterion(output, new_labels.type_as(output), weight=weights).sum(-1).mean()
            else:
                loss = self.criterion(output, new_labels.type_as(output), weight=weights).mean()
            entropy = (output > 0).sum(-1).float().mean()
        else:
            targets_iter = targets_iter.argmax(-1)
            # self.backbone.train()
            loss = F.cross_entropy(output, targets_iter)
            if self.exp_dict["use_gt_baseline"]:
                loss = loss  +  F.cross_entropy(output, labels)
        #loss = self.criterion(output, new_labels).mean()
            entropy = (F.softmax(output) > 0.5).sum(-1).float().mean()
        return loss, entropy

    # def fit(self, epoch, train_loader, val_loader):
    #     score_dict = {}
    #     score_dict.update(self.train_on_loader(epoch, train_loader))
    #     score_dict.update(self.val_on_loader(epoch, val_loader))
    #     return score_dict

    def train_on_loader(self, epoch, train_loader):
        total = 0
        self.backbone.train()
        ret_dict = {'train_teacher_loss': 0, 'train_teacher_entropy_loss': 0}
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            self.optimizer.zero_grad()
            loss, entropy = self.train_on_batch(batch)
            total_loss = loss + entropy*self.exp_dict["entropy_weight"]
            self.backward_and_step(total_loss)
            ret_dict['train_teacher_loss'] += float(loss)
            ret_dict['train_teacher_entropy_loss'] += float(entropy)
            total += len(batch[0])
        return {k: v / total for k, v in ret_dict.items()}

    def train_student_on_loader(self, epoch, train_loader):
        total = 0
        self.backbone.train()
        ret_dict = {'train_student_loss': 0, 'train_student_entropy_loss': 0}
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            self.optimizer.zero_grad()
            loss = self.train_on_preds(batch)
            self.backward_and_step(loss)
            ret_dict['train_student_loss'] += float(loss)
            total += len(batch[0])
        return {k: v / total for k, v in ret_dict.items()}

    def fit(self, epoch, train_loader, val_loader, train_loader_pred=None):
        """Iterate over the training set
        
        Args:
            data_loader (torch.utils.data.DataLoader): a pytorch dataloader
            max_iter (int, optional): Max number of iterations if the end of the dataset is not reached. Defaults to None.
        
        Returns:
            metrics: dictionary with metrics of the training set
        """
        # TODO (random order of priority)
        # 1. Only train on pseudo-labels every x iterations
        #  1.1 pass epoch to train_on_loader
        #  1.2 check if current epoch % a number == 0
        # 2. Different optimizers?
        # 3. EMA (exponential moving average) of the model or the labels
        # 4. Re-read noisy student and chekc if they use any tricks
        # 5. Warm-up for some epochs before starting doing the iterative learning (this can be nicely implemented after 1.1)
        # 6. Smooth one-hot training labels to be between 0.1 and 0.9 instead of 0-1.)
        
        self.backbone.train()
        # Iterate through tasks, each iteration loads n tasks, with n = number of GPU
        
        total = 0
        pseudo_count = 0
        ret_dict_list= []
        if epoch < self.exp_dict["warmup"] or epoch % self.exp_dict["jump"]==0:
            print("Training Teacher...")
            self.backbone.dropout_rate = 0
            
            if self.exp_dict["iterative_copy_weights"] == True:
                self.student_weights = deepcopy(self.backbone.state_dict())
            
            for k  in range(self.exp_dict["k1"]):
                ret_dict = {}
                ret_dict['epoch'] = self.epoch
                ret_dict.update(self.train_on_loader(epoch, train_loader))
                print("Validating..")
                ret_dict.update(self.val_on_loader(epoch, val_loader))
                ret_dict = {k.replace("val_", "val_teacher_"): v for k, v in ret_dict.items()}
                self.teacher_top1_acc = ret_dict["val_teacher_acc_1"]
                ret_dict_list.append(ret_dict)
                self.epoch += 1
        else:
            self.backbone.dropout_rate = 0
            print("Generating pseudolabels...")
            pseudo_count = self.compute_labels(train_loader_pred if train_loader_pred is not None else train_loader)

            print("Restoring student weights")
            self.backbone.dropout_rate = self.exp_dict["dropout_rate"]
            if self.exp_dict["iterative_copy_weights"] == True:
                self.backbone.load_state_dict(self.student_weights)

            for k  in range(self.exp_dict["k2"]):
                ret_dict = {}
                ret_dict['epoch'] = self.epoch

                print("Training on pseudolabels...")
                self.train_student_on_loader(epoch, train_loader)

                print("Validating Student...")
                ret_dict.update(self.val_on_loader(epoch, val_loader))
                ret_dict = {k.replace("val_", "val_student_"): v for k, v in ret_dict.items()}
                ret_dict["diff_top1"] = ret_dict["val_student_acc_1"] - self.teacher_top1_acc
                ret_dict_list.append(ret_dict)
                self.epoch += 1
        return ret_dict_list
    
    @torch.no_grad()
    def compute_labels(self, data_loader):
        self.previous_outputs = []
        self.backbone.eval()
        acc = []
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            idx, images, labels = batch
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            # compute output
            output = self.backbone(images)
            self.previous_outputs.append((torch.sigmoid(output) > self.exp_dict["binary_threshold"]).cpu())            
        self.previous_outputs = torch.cat(self.previous_outputs, 0) # this could be made a moving average (hyperparameter)
        return self.previous_outputs.sum(axis=1).float().mean().item()

    
    def get_state_dict(self):
        """Obtains the state dict of this backbone including optimizer, scheduler, etc
        
        Returns:
            dict: state dict
        """
        ret = {}
        ret['state_dict'] = self.state_dict()
        ret['optimizer'] = self.optimizer.state_dict()
        ret['scheduler'] = self.scheduler.state_dict()
        ret['scaler'] = self.scaler.state_dict()
        try:
            ret['previous_outputs'] = self.previous_outputs
        except:
            pass
        # try:
        #     ret["teacher_weights"] = self.teacher.state_dict()
        # except:
        #     pass
        ret["epoch"] = self.epoch
        try:
            ret["teacher_top1_acc"] = self.teacher_top1_acc
        except:
            pass
        return ret

    def set_state_dict(self, state_dict):
        """Loads the state of the backbone
        
        Args:
            state_dict (dict): The state to load
        """
        self.load_state_dict(state_dict['state_dict'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])
        # self.scaler.load_state_dict(state_dict["scaler"])
        try:
            self.previous_outputs = state_dict["previous_outputs"]
        except:
            pass
        # try:
        #     self.teacher.load_state_dict(state_dict["teacher_weights"])
        # except:
        #     pass
        self.epoch = state_dict["epoch"]
        try:
            self.teacher_top1_acc = state_dict["teacher_top1_acc"]
        except:
            pass