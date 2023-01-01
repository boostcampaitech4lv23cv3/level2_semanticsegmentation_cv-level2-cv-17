import torch
import torch.nn as nn
from torchvision import models


def fcn_resnet50(pretrained, device):
    if type(pretrained) == str:
        model = models.segmentation.fcn_resnet50(pretrained=True)
        model.classifier[4] = nn.Conv2d(512, 11, kernel_size=1)

        checkpoint = torch.load(pretrained, map_location=device)
        model.load_state_dict(checkpoint)
        model = model.to(device)

    elif pretrained:
        model = models.segmentation.fcn_resnet50(pretrained=True)
        model.classifier[4] = nn.Conv2d(512, 11, kernel_size=1)

    else:
        pass

    return model
