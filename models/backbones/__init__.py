import copy
import torchvision.models as models
# from .efficientnet import *
# from .resnet import *
# from .dla_simple import *
import torch.cuda.amp as amp
import numpy as np
import torch.nn as nn
import os
import torchvision
from torchvision.models.resnet import load_state_dict_from_url, model_urls, BasicBlock, Bottleneck
import torch
import torch.nn.functional as F
from typing import Type, Any, Callable, Union, List, Optional
from .resnet_ssl  import get_resnet, name_to_params
from .resnet_152 import get_resnet
# import timm

class ResNet(torchvision.models.ResNet):
    def __init__(self, *args, exp_dict, **kwargs):
        super().__init__(*args, **kwargs)
        self.dropout_rate = exp_dict["dropout_rate"]
        self.pooling = exp_dict["teacher_max_pooling"]
        num_ftrs = self.fc.in_features
        self.fc = nn.Linear(num_ftrs, exp_dict["num_classes"])
        self.exp_dict = exp_dict
        if self.exp_dict.get("sigmoid_way", False) == True \
                and self.exp_dict.get("adjust_bias", False):
            self.fc.bias.data[...] = -np.log(self.exp_dict["num_classes"])

    

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = F.dropout2d(x, self.dropout_rate, training=self.training)
        x = self.layer2(x)
        x = F.dropout2d(x, self.dropout_rate, training=self.training)
        x = self.layer3(x)
        x = F.dropout2d(x, self.dropout_rate, training=self.training)
        x = self.layer4(x)
        # if self.pooling=='gap':
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # elif self.pooling == 'gmp':
        #     x = x.view(x.shape[0], x.shape[1], -1).max(dim=2)[0]
        # else:
        #     raise ValueError(pooling)
        #x = self.avgpool(x)
        
        #return x
        return self.fc(x)

def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def get_model(base_encoder, pretrained=False, progress=True, **kwargs):
    if base_encoder == 'resnet18':
        return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                    **kwargs)
    elif base_encoder == 'resnet34':
        return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                    **kwargs)
    elif base_encoder == 'resnet50':
        return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                    **kwargs)
    elif base_encoder == 'resnet152':
        return get_resnet(depth=152, width_multiplier=3, sk_ratio=1)
        # return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
        #             **kwargs)

                    
class Identity(nn.Module):
    """
    Returns the input without any modification.
    Used to override last layer of pretrained resnet model
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
        
class ResnetWrapper(nn.Module):
    def __init__(
        self, exp_dict: dict, num_classes: int, pretrained: bool = False, freeze_head: bool = False
    ):
        super().__init__()

        self.resnet_model = models.resnet18(pretrained=pretrained)
        num_ftrs = self.resnet_model.fc.in_features
        self.resnet_model.fc = Identity()
        self.fc = nn.Linear(num_ftrs, num_classes)
        self.exp_dict = exp_dict
        if self.exp_dict.get("sigmoid_way", False) == True \
                and self.exp_dict.get("adjust_bias", False):
            self.fc.bias.data[...] = -np.log(self.exp_dict["num_classes"])

    def forward(self, x):
        with amp.autocast(enabled=self.exp_dict["amp"] > 0):
            x = self.resnet_model(x)
            x = self.fc(x)
        return x



def get_backbone(exp_dict):
    if exp_dict['arch']=='resnet18':
        return get_model("resnet18", exp_dict=exp_dict, pretrained=exp_dict['pretrained'] == True)
        #return ResnetWrapper(exp_dict, exp_dict['num_classes'], exp_dict['pretrained'] == True)
        #return ResNet18()
    elif "ssl" in exp_dict and exp_dict['arch']=='resnet50' and exp_dict['ssl']=='simclr' and os.path.isfile(exp_dict["pretrained"]):
        model, _ = get_resnet(*name_to_params(exp_dict["pretrained"]))
        model.load_state_dict(torch.load(exp_dict["pretrained"])['resnet'])
        print("=> loaded pre-trained SimCLR  model '{}'".format(exp_dict["pretrained"]))
            #model.load_state_dict(weights["state_dict"])
        return model
    elif exp_dict['arch']=='resnet50':
        model = get_model("resnet50", exp_dict=exp_dict, pretrained=exp_dict['pretrained'] == True)
        if os.path.isfile(exp_dict["pretrained"]):
            loc = 'cuda:{}'.format(0)
            checkpoint = torch.load(exp_dict["pretrained"],map_location=loc)
            state_dict = checkpoint['state_dict']
            
            # checkpoint = torch.load(exp_dict["pretrained"],map_location=loc)
            # state_dict = checkpoint['state_dict']
            # for k in list(state_dict.keys()):
            #     # retain only encoder_q up to before the embedding layer
            #     if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            #         # remove prefix
            #         state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            #     # delete renamed or unused k
            #     del state_dict[k]

            # msg = model.load_state_dict(state_dict, strict=False)
            #assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
            for k in list(state_dict.keys()):
                # retain only encoder_q up to before the embedding layer
                if k.startswith('teacher.module'):
                    del state_dict[k]
                else:
                    state_dict[k[16:]] = state_dict[k]
                # delete renamed or unused k
                    del state_dict[k]
            model.load_state_dict(state_dict)
            print("=> loaded pre-trained model '{}'".format(exp_dict["pretrained"]))
            #model.load_state_dict(weights["state_dict"])
        return model
    elif exp_dict['arch']=='resnext50_32x4d':
        return models.resnext50_32x4d(pretrained=exp_dict['pretrained'] == True)
    elif exp_dict['arch'] == 'resnet50d':
        model = timm.create_model('resnet50d', pretrained=exp_dict['pretrained'])
        return model