import torch
import torch.nn as nn

from models.fcn_resnet50 import fcn_resnet50


def _define_criterion():
    """Loss 함수 변경 시 수정"""
    return nn.CrossEntropyLoss()


def _define_optimizer(model, lr):
    """Optimizer 변경 시 수정"""
    return torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=1e-6)


def _define_model(pretrained, device):
    """Backbone 변경 시 수정"""
    return fcn_resnet50(pretrained, device)


def define_network(pretrained, device, learning_rate=1e-4):
    """Function defining a network for easy customization"""

    model = _define_model(pretrained, device)
    criterion = _define_criterion()
    optimizer = _define_optimizer(model, learning_rate)

    return model, criterion, optimizer
