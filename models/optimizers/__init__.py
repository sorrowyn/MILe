from functools import partial
import torch
from torch.optim.lr_scheduler import MultiStepLR

def get_optimizer(exp_dict):
    if exp_dict['optimizer'] == "sgd":
        return partial(torch.optim.SGD, nesterov=True, momentum=0.9)
    elif exp_dict['optimizer'] == 'adam':
        return torch.optim.AdamW

def get_scheduler(exp_dict):
    if "scheduler" not in exp_dict:
        return torch.optim.lr_scheduler.ReduceLROnPlateau
        #return partial(MultiStepLR, milestones=[exp_dict["max_epoch"] + 1], gamma=0.1)
    else:
        scheduler_dict = exp_dict['scheduler']

    if scheduler_dict["name"] == "step":
        return partial(MultiStepLR, milestones=scheduler_dict["steps"], 
                                    gamma=scheduler_dict.get('gamma', 0.1))
    else:
        raise NotImplementedError

