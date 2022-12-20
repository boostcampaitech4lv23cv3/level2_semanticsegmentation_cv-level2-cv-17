import torch
import torch.nn as nn
from torchvision import models

device = "cuda" if torch.cuda.is_available() else "cpu"


def _define_criterion():
    """Loss 함수 변경 시 수정"""
    return nn.CrossEntropyLoss()


def _define_optimizer(model, lr):
    """Optimizer 변경 시 수정"""
    return torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=1e-6)


def _define_model(pretrained):
    """Backbone 변경 시 수정"""
    if type(pretrained) == str:
        model = models.segmentation.fcn_resnet50(pretrained=True)
        model.classifier[4] = nn.Conv2d(512, 11, kernel_size=1)

        checkpoint = torch.load(pretrained, map_location=device)
        state_dict = checkpoint.state_dict()
        model.load_state_dict(state_dict)
        model = model.to(device)

    elif not pretrained:
        model = models.segmentation.fcn_resnet50(pretrained=False)
        model.classifier[4] = nn.Conv2d(512, 11, kernel_size=1)

    else:
        model = models.segmentation.fcn_resnet50(pretrained=True)
        model.classifier[4] = nn.Conv2d(512, 11, kernel_size=1)

    return model


def define_network(pretrained=True, learning_rate=1e-4):
    """Function defining a network for easy customization"""

    model = _define_model(pretrained)
    criterion = _define_criterion()
    optimizer = _define_optimizer(model, learning_rate)

    return model, criterion, optimizer
