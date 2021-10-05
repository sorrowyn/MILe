import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbones import get_backbone
from .optimizers import get_optimizer, get_scheduler
import torch.optim 
from tqdm import tqdm
import torch.cuda.amp as amp
from .supervised import Model as Supervised
from sklearn.metrics import f1_score

class SoftTargetCrossEntropy(nn.Module):
    def forward(self, x, target):
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()

def f1(outputs, labels, threshold=0.5, sigmoid_way=True):
    #threshold=0.5
    if sigmoid_way:
        outputs = torch.sigmoid(outputs)
        outputs_1 = F.one_hot(outputs.argmax(-1), outputs.shape[-1]).bool()
        outputs_2 = outputs > threshold
        outputs = outputs_1 | outputs_2
    else:
        outputs = F.one_hot(outputs.argmax(-1), outputs.shape[-1]).bool()
    outPRED = outputs.cpu().numpy()
    outGT = labels.cpu().numpy()
    return f1_score(outGT > 0, outPRED, average="samples")

class Model(torch.nn.Module):
    """Trains a backbone using an episodic scheme on multiple GPUs"""

    def __init__(self, exp_dict):
        """ Constructor
        Args:
            backbone: architecture to train
            nclasses: number of output classes
            exp_dict: reference to dictionary with the hyperparameters
        """
        super().__init__()
        #self.backbone = backbone
        self.exp_dict = exp_dict 
        self.lr = exp_dict['lr']
        self.rescale_sigmoid = exp_dict["rescale_sigmoid"]
        # load backbone
        self.backbone = get_backbone(exp_dict).cuda()
        self.backbone = torch.nn.DataParallel(self.backbone, list(range(self.exp_dict["ngpu"]))) 

        self.optimizer = get_optimizer(exp_dict)(self.backbone.parameters(), lr=exp_dict['lr'],
                                                 weight_decay=exp_dict['decay'])
        if "lr2" in exp_dict:
            model_parameters = set(self.backbone.parameters())-set(self.backbone.module.fc.parameters())
            assert(set(self.backbone.parameters()) != model_parameters)
            self.optimizer = get_optimizer(exp_dict)([{"params":list(model_parameters),"lr": exp_dict["lr2"]}, {"params": self.backbone.module.fc.parameters(),"lr": exp_dict["lr"]}], lr=exp_dict['lr'],
                                                 weight_decay=exp_dict['decay'])
        self.scheduler = get_scheduler(exp_dict)(self.optimizer)
        if exp_dict["sigmoid_way"]:
            self.sigmoid_way = True
            self.criterion = lambda x,y,weight: nn.BCEWithLogitsLoss(reduction='none', weight=weight.unsqueeze(-1))(x,y)
        else:
            self.sigmoid_way = False
            self.criterion = lambda x,y,weight: nn.CrossEntropyLoss(weight=weight)(x,y)

        if self.exp_dict["amp"] > 0:
            self.scaler = amp.GradScaler()

    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k)
            return tuple(res)

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
        else:
            if self.exp_dict["label_smoothing"] > 0:
                labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=self.exp_dict["num_classes"])
                labels_one_hot = labels_one_hot * self.exp_dict["label_smoothing"] + (1 - self.exp_dict['label_smoothing']) / self.exp_dict["num_classes"]
                loss = F.cross_entropy(output, labels_one_hot)
            else:
                loss = F.cross_entropy(output, labels)
        return loss

    @torch.no_grad()
    def val_on_batch(self, batch):
        """Computes the loss and accuracy on a validation batch
        
        Args:
            batch (dict): Episode dict
        
        Returns:
            tuple: loss and accuracy of the episode 
        """
        paths, images, labels, real_labels = batch
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels_one_hot = real_labels.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        # compute output
        output = self.backbone(images)
        outputs = torch.sigmoid(output)
        outputs_1 = F.one_hot(outputs.argmax(-1), outputs.shape[-1]).bool()
        outputs_2 = outputs > self.exp_dict["binary_threshold"]
        outputs = outputs_1 | outputs_2
        if self.sigmoid_way:
            tw = labels_one_hot*(self.exp_dict['num_classes'] - 1) + 1.0
            loss = (self.criterion(output, labels_one_hot.type_as(output), weight=torch.tensor([1.0]).cuda())*tw).mean()
        else:
            loss = F.cross_entropy(output, labels)
            #loss = self.criterion(output, labels, weight=torch.tensor([1.0]).cuda())
        real_accuracy = labels_one_hot[torch.arange(labels_one_hot.size(0)), output.argmax(-1)].sum().item()
        real_accuracy_05 = ((labels_one_hot>0) & outputs).any(-1).sum()
        accuracy_1, accuracy_5 = self.accuracy(output, labels, topk=(1, 5))
        real_accuracy_f1 = f1(output, labels_one_hot, self.exp_dict["binary_threshold"], sigmoid_way=self.sigmoid_way)
        total_real = (labels_one_hot > 0).any(-1).sum()
        real_accuracy_f1_05 = f1(output, labels_one_hot, 0.5, sigmoid_way=self.sigmoid_way)
        real_accuracy_f1_025 = f1(output, labels_one_hot, 0.25, sigmoid_way=self.sigmoid_way)
        return loss, accuracy_1, accuracy_5, real_accuracy, real_accuracy_05, real_accuracy_f1, real_accuracy_f1_05, real_accuracy_f1_025, float(total_real)

    @torch.no_grad()
    def val_on_batch_wv(self, batch):
        """Computes the loss and accuracy on a validation batch
        
        Args:
            batch (dict): Episode dict
        
        Returns:
            tuple: loss and accuracy of the episode 
        """
        paths, images, labels = batch
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            #labels_one_hot = F.one_hot(labels, 1000)
            #labels_one_hot = labels_one_hot.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        # compute output
        output = self.backbone(images)
        if self.sigmoid_way:
            labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=self.exp_dict["num_classes"])
            if self.rescale_sigmoid:
                # tw = labels_one_hot*(self.exp_dict['num_classes'] - 1) + 1.0
                # loss = (self.criterion(output, labels_one_hot.type_as(output))*tw).mean()
                tw = labels_one_hot*(self.exp_dict['num_classes'] - 1) + 1.0
                # loss = (self.criterion(output, labels_one_hot.type_as(output))*tw).mean()
                loss = (self.criterion(output, labels_one_hot.type_as(output), weight=torch.tensor([1.0]).cuda())*tw).mean()
                #loss = self.criterion(output, labels_one_hot.type_as(output), weight=torch.tensor([1.0]).cuda()).sum(-1).mean()
            else:
                loss = (self.criterion(output, labels_one_hot.type_as(output))).mean()
        else:
            loss = F.cross_entropy(output, labels)

        # if self.sigmoid_way:
        #     tw = labels_one_hot*(self.exp_dict['num_classes'] - 1) + 1.0
        #     loss = (self.criterion(output, labels_one_hot.type_as(output))*tw).mean()
        # else:
        #     loss = self.criterion(output, labels)
        
        accuracy_1, accuracy_5 = self.accuracy(output, labels, topk=(1, 5))
        real_accuracy = labels_one_hot[torch.arange(labels_one_hot.size(0)), output.argmax(-1)].sum().item()
        total_real = (labels_one_hot > 0).any(-1).sum()
        return loss, accuracy_1, accuracy_5, real_accuracy, float(total_real)


    def train_on_loader(self, epoch, data_loader, max_iter=None, debug_plot_path=None):
        """Iterate over the training set
        
        Args:
            data_loader (torch.utils.data.DataLoader): a pytorch dataloader
            max_iter (int, optional): Max number of iterations if the end of the dataset is not reached. Defaults to None.
        
        Returns:
            metrics: dictionary with metrics of the training set
        """
        self.backbone.train()
        # Iterate through tasks, each iteration loads n tasks, with n = number of GPU
        ret_dict = {'loss': 0.}
        total = 0
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            self.optimizer.zero_grad()
            loss = self.train_on_batch(batch)
            self.backward_and_step(loss)
            ret_dict['loss'] += float(loss)
            total += len(batch[0])
        return {"train_loss": ret_dict['loss'] / len(data_loader)}
        

    @torch.no_grad()
    def val_on_loader(self, epoch, data_loader, max_iter=None):
        """Iterate over the validation set

        Args:
            data_loader: iterable validation data loader
            max_iter: max number of iterations to perform if the end of the dataset is not reached
        """
        self.backbone.eval()
        ret_dict = {'val_loss': 0., 'val_acc': 0., 'val_real_acc': 0., 'val_real_acc_05': 0., 'val_acc_1': 0., 'val_acc_5': 0., 'val_real_acc_f1': 0.,
                    'val_real_acc_f1_05': 0.,'val_real_acc_f1_025': 0.}
        total = 0
        iters = 0
        total_reals = 0
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            if max_iter is not None and batch_idx >= max_iter:
                break
            loss, acc_1, acc_5, real_acc, real_acc_05, real_acc_f1, real_acc_f1_05, real_acc_f1_025, total_real = self.val_on_batch(batch)
            ret_dict['val_loss'] += float(loss)
            ret_dict['val_acc_1'] += float(acc_1)
            ret_dict['val_acc_5'] += float(acc_5)
            ret_dict['val_real_acc'] += float(real_acc)
            ret_dict['val_real_acc_05'] += float(real_acc_05)
            ret_dict['val_real_acc_f1'] += float(real_acc_f1) * len(batch[0])
            ret_dict['val_real_acc_f1_05'] += float(real_acc_f1_05) * len(batch[0])
            ret_dict['val_real_acc_f1_025'] += float(real_acc_f1_025) * len(batch[0])
            total += len(batch[0])
            total_reals += total_real
            iters += 1

        ret_dict['val_loss'] = ret_dict['val_loss']/iters
        ret_dict['val_acc_1'] = ret_dict['val_acc_1']/total
        ret_dict['val_acc_5'] = ret_dict['val_acc_5']/total
        ret_dict['val_real_acc'] = ret_dict['val_real_acc']/total_reals
        ret_dict['val_real_acc_05'] = ret_dict['val_real_acc_05']/total_reals
        ret_dict['val_real_acc_f1'] = ret_dict['val_real_acc_f1']/total_reals
        ret_dict['val_real_acc_f1_05'] = ret_dict['val_real_acc_f1_05']/total_reals
        ret_dict['val_real_acc_f1_025'] = ret_dict['val_real_acc_f1_025']/total_reals
        if epoch % self.exp_dict["validation_iters"]==0:
            if "step" in self.exp_dict.get("scheduler", {"name": ""})["name"]:
                self.scheduler.step()
            else:
                self.scheduler.step(ret_dict['val_loss'])

        return ret_dict

    @torch.no_grad()
    def val_on_loader_wv(self, epoch, data_loader, max_iter=None):
        """Iterate over the validation set

        Args:
            data_loader: iterable validation data loader
            max_iter: max number of iterations to perform if the end of the dataset is not reached
        """
        self.backbone.eval()
        ret_dict = {'val_loss_wv': 0., 'val_acc_wv': 0., 'val_real_acc_wv': 0., 'val_acc_1_wv': 0., 'val_acc_5_wv': 0.}
        total = 0
        iters = 0
        total_reals_wv = 0
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            if max_iter is not None and batch_idx >= max_iter:
                break
            loss, acc_1, acc_5, real_acc, total_real = self.val_on_batch_wv(batch)
            ret_dict['val_loss_wv'] += float(loss)
            ret_dict['val_acc_1_wv'] += float(acc_1)
            ret_dict['val_acc_5_wv'] += float(acc_5)
            ret_dict['val_real_acc_wv'] += float(real_acc)
            total += len(batch[0])
            total_reals_wv += total_real
            iters += 1

        ret_dict['val_loss_wv'] = ret_dict['val_loss_wv']/iters
        ret_dict['val_acc_1_wv'] = ret_dict['val_acc_1_wv']/total
        ret_dict['val_acc_5_wv'] = ret_dict['val_acc_5_wv']/total
        ret_dict['val_real_acc_wv'] = ret_dict['val_real_acc_wv']/total_reals_wv
        
        if "step" in self.exp_dict.get("scheduler", {"name": ""})["name"]:
            self.scheduler.step()
        else:
            self.scheduler.step(ret_dict['val_loss_wv'])

        return ret_dict

    @torch.no_grad()
    def test_on_batch(self, batch):
        """Computes the loss and accuracy on a validation batch
        
        Args:
            batch (dict): Episode dict
        
        Returns:
            tuple: loss and accuracy of the episode 
        """
        paths, images, labels, real_labels = batch
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels_one_hot = real_labels.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        # compute output
        output = self.backbone(images)
        if self.sigmoid_way:
            tw = labels_one_hot*(self.exp_dict['num_classes'] - 1) + 1.0
            loss = (self.criterion(output, labels_one_hot.type_as(output), weight=torch.tensor([1.0]).cuda())*tw).mean()
        else:
            loss = self.criterion(output, labels)
        #TODO: output multi label predictions
        #output real labels
        #return path of image or image itself to visualize in notebook (for the mistakes)
        #modify dataloader to output paths for the last option
        #create a notebook
        if "binary_threshold" not in self.exp_dict:
            self.exp_dict["binary_threshold"] = 0.5
        real_accuracy = labels_one_hot[torch.arange(labels_one_hot.size(0)), output.argmax(-1)].sum().item()
        real_accuracy_f1 = f1( output,labels_one_hot, self.exp_dict["binary_threshold"], sigmoid_way=self.sigmoid_way)
        real_accuracy_f1_05 = f1( output,labels_one_hot, 0.5, sigmoid_way=self.sigmoid_way)
        real_accuracy_f1_025 = f1( output,labels_one_hot, 0.25, sigmoid_way=self.sigmoid_way)
        accuracy_1, accuracy_5 = self.accuracy(output, labels, topk=(1, 5))
        total_real = (labels_one_hot > 0).any(-1).sum()
        return loss, accuracy_1, accuracy_5, real_accuracy, real_accuracy_f1, real_accuracy_f1_05, real_accuracy_f1_025, real_labels, labels, output, paths, float(total_real)

    @torch.no_grad()
    def test_on_loader(self, epoch, data_loader, max_iter=None):
        """Iterate over the validation set

        Args:
            data_loader: iterable validation data loader
            max_iter: max number of iterations to perform if the end of the dataset is not reached
        """
        self.backbone.eval()
        ret_dict = {'val_loss': 0., 'val_acc': 0., 'val_real_acc': 0., 'val_acc_1': 0., 'val_acc_5': 0.,
                    'labels': [], 'real_labels': [], 'output': [], 'paths': [], 'val_real_acc_f1': 0.,
                    'val_real_acc_f1_05': 0.,'val_real_acc_f1_025': 0.}
        total = 0
        iters = 0
        total_reals = 0
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            if max_iter is not None and batch_idx >= max_iter:
                break
            loss, accuracy_1, accuracy_5, real_accuracy, real_accuracy_f1, real_accuracy_f1_05, real_accuracy_f1_025, real_labels, labels, output, paths, total_real = self.test_on_batch(batch)
            ret_dict['val_loss'] += float(loss)
            ret_dict['val_acc_1'] += float(accuracy_1)
            ret_dict['val_acc_5'] += float(accuracy_5)
            ret_dict['val_real_acc'] += float(real_accuracy)
            ret_dict['real_labels']  += [real_labels.data.cpu().numpy()]
            ret_dict['labels'] += [labels.data.cpu().numpy()]
            ret_dict['output'] += [output.data.cpu().numpy()]
            ret_dict['val_real_acc_f1'] += float(real_accuracy_f1) * len(batch[0])
            ret_dict['val_real_acc_f1_05'] += float(real_accuracy_f1_05) * len(batch[0])
            ret_dict['val_real_acc_f1_025'] += float(real_accuracy_f1_025) * len(batch[0])
            ret_dict['paths'] += list(paths)

            total += len(batch[0])
            total_reals += total_real
            iters += 1
            

        ret_dict['val_loss'] = ret_dict['val_loss']/iters
        ret_dict['val_acc_1'] = ret_dict['val_acc_1']/total
        ret_dict['val_acc_5'] = ret_dict['val_acc_5']/total
        ret_dict['val_real_acc'] = ret_dict['val_real_acc']/total_reals
        ret_dict['val_real_acc_f1'] = ret_dict['val_real_acc_f1']/total_reals
        ret_dict['val_real_acc_f1_05'] = ret_dict['val_real_acc_f1_05']/total_reals
        ret_dict['val_real_acc_f1_025'] = ret_dict['val_real_acc_f1_025']/total_reals
        ret_dict['real_labels'] = np.concatenate(ret_dict['real_labels'],0)
        ret_dict['labels']= np.concatenate(ret_dict['labels'],0)
        ret_dict['output']= np.concatenate(ret_dict['output'],0)
        ret_dict['paths']= ret_dict['paths']
        return ret_dict
  
    def fit(self, epoch, train_loader, val_loader):
        score_dict = {}
        score_dict.update(self.train_on_loader(epoch, train_loader))
        score_dict.update(self.val_on_loader(epoch, val_loader))
        return score_dict

    def get_lr(self):
        return self.lr

    def backward_and_step(self, loss):
        if self.exp_dict["amp"] > 0:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

    def get_state_dict(self):
        """Obtains the state dict of this backbone including optimizer, scheduler, etc
        
        Returns:
            dict: state dict
        """
        ret = {}
        ret['state_dict'] = self.state_dict()
        ret['optimizer'] = self.optimizer.state_dict()
        ret['scheduler'] = self.scheduler.state_dict()
        return ret

    def set_state_dict(self, state_dict):
        """Loads the state of the backbone
        
        Args:
            state_dict (dict): The state to load
        """
        self.load_state_dict(state_dict['state_dict'])
        if "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.scheduler.load_state_dict(state_dict['scheduler'])
        